"""Fixtures for the API integration tests.

Postgres is real, not SQLite: the ingestion and bandit paths use
`ON CONFLICT DO UPDATE`, which SQLite does not express the same way. Testing
against a different dialect than production would mean the one thing these
tests exist to check — that the SQL is right — is exactly what they don't check.

Qdrant is faked. It is a pure similarity lookup with no application logic worth
testing through a network hop, and faking it keeps the suite fast enough to run
on every save.
"""
import os
import uuid
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

DB_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://mde:mde@localhost:5432/mde"
)


def _postgres_available() -> bool:
    try:
        engine = create_engine(DB_URL, connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


requires_postgres = pytest.mark.skipif(
    not _postgres_available(), reason="Postgres not reachable — start docker compose"
)


class FakeVectorStore:
    """In-memory stand-in for Qdrant with the same surface the app uses."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.vectors: dict[str, np.ndarray] = {}
        self.payloads: dict[str, dict[str, Any]] = {}
        self.search_calls: list[dict[str, Any]] = []

    def ensure_collection(self) -> None:
        pass

    def upsert(self, records) -> None:
        for track_id, vector, payload in records:
            self.vectors[track_id] = np.asarray(vector, dtype=np.float32)
            self.payloads[track_id] = payload

    def get_vector(self, track_id: str):
        return self.vectors.get(track_id)

    def search(
        self,
        vector,
        *,
        limit: int = 200,
        exclude_track_ids: list[str] | None = None,
        tags: list[str] | None = None,
        tempo_range: tuple[float, float] | None = None,
        with_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        # Recorded so tests can assert the service asked for vectors only when
        # the selected arm actually diversifies.
        self.search_calls.append({"with_vectors": with_vectors, "limit": limit})

        excluded = set(exclude_track_ids or [])
        hits = []
        for track_id, stored in self.vectors.items():
            if track_id in excluded:
                continue
            payload = self.payloads.get(track_id, {})
            if tags and not (set(tags) & set(payload.get("tags") or [])):
                continue
            if tempo_range:
                tempo = payload.get("tempo_bpm")
                if tempo is None or not (tempo_range[0] <= tempo <= tempo_range[1]):
                    continue

            candidate = {
                **payload,
                "track_id": track_id,
                "audio_score": float(np.asarray(vector, dtype=np.float32) @ stored),
            }
            if with_vectors:
                candidate["_vector"] = stored
            hits.append(candidate)

        hits.sort(key=lambda h: h["audio_score"], reverse=True)
        return hits[:limit]

    def count(self) -> int:
        return len(self.vectors)


@pytest.fixture
def store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def db_session(schema):  # noqa: ARG001 - ordering dependency
    from app.db import SessionLocal

    with SessionLocal() as session:
        yield session


@pytest.fixture(scope="session")
def schema():
    """Create tables once per session against the real database."""
    if not _postgres_available():
        pytest.skip("Postgres not reachable")

    os.environ.setdefault("DATABASE_URL", DB_URL)
    from app.db import Base, engine

    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def clean_tables(schema):
    """Truncate between tests so arms and events start from a known state."""
    engine = schema
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE interaction_events, bandit_arms, tracks CASCADE"))
    yield
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE interaction_events, bandit_arms, tracks CASCADE"))


@pytest.fixture
def client(schema, clean_tables, store, monkeypatch):  # noqa: ARG001
    """TestClient with the fake vector store and the cache disabled.

    The cache is off because these tests assert on ranking behaviour; a warm
    entry from a previous test would make them pass without exercising
    anything.
    """
    from app import cache
    from app.api.deps import get_store
    from app.main import app

    monkeypatch.setattr(cache, "_client", lambda: None)
    app.dependency_overrides[get_store] = lambda: store

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def seeded(client, store):
    """Three orthogonal clusters, arranged so each behaviour is observable.

    Every arrangement here is load-bearing:

    * **Cluster A** gives each track its own artist, so the per-artist cap
      cannot dictate the ordering and the ranking signals are what actually
      decide. `a4` is both the least popular and the furthest away, which is
      the only configuration where the novelty weight is falsifiable.
    * **Cluster C** puts four tracks under one artist, so the cap has three
      non-seed candidates to actually cap.
    * Jitter lives on a dimension no cluster uses, so it perturbs distance
      without pulling a track toward another cluster.
    """
    from app.db import SessionLocal
    from app.models import Track

    dim = store.dim
    cluster_a = np.zeros(dim, dtype=np.float32)
    cluster_a[0] = 1.0
    cluster_b = np.zeros(dim, dtype=np.float32)
    cluster_b[1] = 1.0
    cluster_c = np.zeros(dim, dtype=np.float32)
    cluster_c[2] = 1.0

    specs = [
        # (id, artist, cluster, jitter, tags, listens)
        ("a1", "artist-1", cluster_a, 0.00, ["rock"], 1_000_000),
        ("a2", "artist-2", cluster_a, 0.02, ["rock"], 500_000),
        ("a3", "artist-3", cluster_a, 0.04, ["rock"], 300_000),
        # Same tags as its cluster on purpose: a differing tag set would change
        # the tag signal too, and the novelty test could no longer attribute a
        # rank change to novelty alone.
        ("a4", "artist-4", cluster_a, 0.05, ["rock"], 5),
        ("b1", "artist-5", cluster_b, 0.00, ["jazz"], 900),
        ("b2", "artist-5", cluster_b, 0.03, ["jazz", "piano"], 3),
        ("c1", "artist-6", cluster_c, 0.00, ["ambient"], 100),
        ("c2", "artist-6", cluster_c, 0.01, ["ambient"], 100),
        ("c3", "artist-6", cluster_c, 0.02, ["ambient"], 100),
        ("c4", "artist-6", cluster_c, 0.03, ["ambient"], 100),
    ]

    records = []
    with SessionLocal() as db:
        for track_id, artist, centroid, jitter, tags, listens in specs:
            vector = centroid.copy()
            vector[-1] = jitter  # unused dimension: perturbs distance only
            vector /= np.linalg.norm(vector)

            features = {"tempo_bpm": 120.0, "loudness_db": -10.0}
            db.merge(
                Track(
                    id=track_id,
                    name=f"Track {track_id}",
                    artist_id=artist,
                    artist_name=artist,
                    audio_url=f"https://example.invalid/{track_id}.mp3",
                    duration_sec=180,
                    tags=tags,
                    features=features,
                    embedded=True,
                )
            )
            payload = {
                "artist_id": artist,
                "tags": tags,
                "listens": listens,
                **features,
            }
            records.append((track_id, vector, payload))
        db.commit()

    store.upsert(records)
    return [s[0] for s in specs]


@pytest.fixture
def session_id() -> str:
    return str(uuid.uuid4())
