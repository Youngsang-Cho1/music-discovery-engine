"""Qdrant-backed ANN index over CLAP audio embeddings.

This is stage 1 of the two-stage recommender. v1 scored the *entire* catalog on
every request with a full NumPy scan — fine at 16k tracks, linear and hopeless
past that. HNSW gives sublinear candidate generation, and payload filters push
genre/tempo constraints into the index instead of masking scores afterwards.
"""
import logging
import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from app.config import get_settings

logger = logging.getLogger(__name__)


def _point_id(track_id: str) -> str:
    """Qdrant point ids must be UUID or unsigned int; Jamendo ids are strings.

    A deterministic UUID5 keeps upserts idempotent — re-ingesting a track
    overwrites its point instead of duplicating it.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"jamendo:track:{track_id}"))


class VectorStore:
    def __init__(self, url: str | None = None, collection: str | None = None):
        settings = get_settings()
        self.collection = collection or settings.qdrant_collection
        self.dim = settings.embedding_dim
        self.client = QdrantClient(url=url or settings.qdrant_url, timeout=30)

    def ensure_collection(self) -> None:
        """Create the collection and payload indexes if absent (idempotent)."""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                # CLAP embeddings are L2-normalized, so cosine == dot product.
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            logger.info("created qdrant collection %s", self.collection)

        # Payload indexes make filtered search use the index rather than a scan.
        for field, schema in (
            ("track_id", PayloadSchemaType.KEYWORD),
            ("artist_id", PayloadSchemaType.KEYWORD),
            ("tags", PayloadSchemaType.KEYWORD),
            ("tempo_bpm", PayloadSchemaType.FLOAT),
        ):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:  # already indexed
                pass

    def upsert(self, records: list[tuple[str, np.ndarray, dict[str, Any]]]) -> None:
        """Upsert (track_id, embedding, payload) triples."""
        if not records:
            return
        points = [
            PointStruct(
                id=_point_id(track_id),
                vector=vector.astype(np.float32).tolist(),
                payload={"track_id": track_id, **payload},
            )
            for track_id, vector, payload in records
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def get_vector(self, track_id: str) -> np.ndarray | None:
        records = self.client.retrieve(
            collection_name=self.collection,
            ids=[_point_id(track_id)],
            with_vectors=True,
        )
        if not records or records[0].vector is None:
            return None
        return np.asarray(records[0].vector, dtype=np.float32)

    def search(
        self,
        vector: np.ndarray,
        *,
        limit: int = 200,
        exclude_track_ids: list[str] | None = None,
        tags: list[str] | None = None,
        tempo_range: tuple[float, float] | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search returning candidate payloads with their cosine scores."""
        must: list[Any] = []
        must_not: list[Any] = []

        if tags:
            must.append(FieldCondition(key="tags", match=MatchAny(any=tags)))
        if tempo_range:
            low, high = tempo_range
            must.append(FieldCondition(key="tempo_bpm", range=Range(gte=low, lte=high)))
        if exclude_track_ids:
            must_not.append(
                FieldCondition(key="track_id", match=MatchAny(any=exclude_track_ids))
            )

        query_filter = (
            Filter(must=must or None, must_not=must_not or None)
            if (must or must_not)
            else None
        )

        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector.astype(np.float32).tolist(),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        ).points

        return [{**(h.payload or {}), "audio_score": float(h.score)} for h in hits]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count
