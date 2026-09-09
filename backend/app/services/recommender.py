"""Recommendation service: the two-stage pipeline behind the API.

    seed track / text query
            │
            ▼
    CLAP vector  ──►  Qdrant HNSW (stage 1)  ──►  ~200 candidates
                                                       │
                                                       ▼
                                     weighted signal fusion + MMR (stage 2)
                                                       │
                                                       ▼
                                                    top-K
"""
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings.descriptors import DESCRIPTOR_KEYS
from app.experiments.variants import get_variant
from app.models import Track
from app.retrieval.rerank import RankingWeights, mmr_select, score_candidates
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

# How many ANN candidates to pull before reranking. Large enough that MMR and
# the artist cap have something to choose from, small enough to rerank in
# single-digit milliseconds.
CANDIDATE_POOL = 200


@dataclass(slots=True)
class Recommendation:
    track: Track
    relevance: float
    signals: dict[str, float]


def _track_payload(track: Track) -> dict[str, Any]:
    """Flatten a Track row into the dict shape the reranker expects."""
    features = track.features or {}
    return {
        "track_id": track.id,
        "artist_id": track.artist_id,
        "tags": track.tags or [],
        **{k: features.get(k) for k in DESCRIPTOR_KEYS if k in features},
    }


def _load_tracks(db: Session, track_ids: list[str]) -> dict[str, Track]:
    if not track_ids:
        return {}
    rows = db.execute(select(Track).where(Track.id.in_(track_ids))).scalars()
    return {t.id: t for t in rows}


def recommend_by_track(
    db: Session,
    store: VectorStore,
    *,
    seed_track_id: str,
    variant: str,
    top_k: int = 20,
    tags: list[str] | None = None,
    tempo_range: tuple[float, float] | None = None,
) -> list[Recommendation]:
    """Seed-track recommendations."""
    seed = db.get(Track, seed_track_id)
    if seed is None:
        return []

    seed_vector = store.get_vector(seed_track_id)
    if seed_vector is None:
        logger.warning("track %s has no embedding", seed_track_id)
        return []

    weights = get_variant(variant)
    candidates = store.search(
        seed_vector,
        limit=CANDIDATE_POOL,
        exclude_track_ids=[seed_track_id],
        tags=tags,
        tempo_range=tempo_range,
        with_vectors=weights.mmr_lambda < 1.0,
    )
    return _finish(db, candidates, _track_payload(seed), weights, top_k)


def recommend_by_text(
    db: Session,
    store: VectorStore,
    *,
    query: str,
    variant: str,
    top_k: int = 20,
    tags: list[str] | None = None,
    tempo_range: tuple[float, float] | None = None,
) -> list[Recommendation]:
    """Natural-language search — impossible in v1, free with CLAP.

    The text encoder writes into the same space as the audio encoder, so the
    query vector goes straight into the same HNSW index with no separate
    text index and no keyword matching against track titles.
    """
    from app.embeddings.clap import embed_text

    weights = get_variant(variant)
    query_vector = embed_text([query])[0]
    candidates = store.search(
        query_vector,
        limit=CANDIDATE_POOL,
        tags=tags,
        tempo_range=tempo_range,
        with_vectors=weights.mmr_lambda < 1.0,
    )
    # No seed track: descriptor and tag similarity have nothing to compare
    # against, so relevance falls back to the audio-text alignment.
    return _finish(db, candidates, {}, weights, top_k)


def _finish(
    db: Session,
    candidates: list[dict[str, Any]],
    seed_payload: dict[str, Any],
    weights: RankingWeights,
    top_k: int,
) -> list[Recommendation]:
    if not candidates:
        return []

    scored = score_candidates(seed_payload, candidates, weights)

    # Vectors ride along from the ANN query when the variant diversifies, so
    # MMR costs no extra requests.
    vectors: dict[str, np.ndarray] = {
        cand["track_id"]: cand["_vector"] for cand in scored if "_vector" in cand
    }

    selected = mmr_select(scored, vectors, top_k=top_k, mmr_lambda=weights.mmr_lambda)

    tracks = _load_tracks(db, [c["track_id"] for c in selected])
    out: list[Recommendation] = []
    for cand in selected:
        track = tracks.get(cand["track_id"])
        if track is None:
            continue
        out.append(
            Recommendation(
                track=track,
                relevance=cand["relevance"],
                signals=cand["signals"],
            )
        )
    return out
