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

from app import cache
from app.embeddings.descriptors import DESCRIPTOR_KEYS
from app.embeddings.text_encoder import encode_query
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


def _ranking_cache_key(
    *,
    seed: str | None,
    query: str | None,
    variant: str,
    top_k: int,
    tags: list[str] | None,
    tempo_range: tuple[float, float] | None,
) -> str:
    return cache.make_key(
        "rank",
        seed=seed,
        query=query,
        variant=variant,
        top_k=top_k,
        tags=sorted(tags) if tags else None,
        tempo=list(tempo_range) if tempo_range else None,
    )


def _hydrate(db: Session, ranked: list[dict[str, Any]]) -> list[Recommendation]:
    """Turn a ranked [{track_id, relevance, signals}] list into Recommendations.

    Track rows are always read fresh from Postgres rather than cached with the
    ranking, so a title or artwork fix shows up immediately instead of waiting
    out the ranking TTL.
    """
    tracks = _load_tracks(db, [r["track_id"] for r in ranked])
    out: list[Recommendation] = []
    for row in ranked:
        track = tracks.get(row["track_id"])
        if track is None:
            continue  # track deleted since the ranking was cached
        out.append(
            Recommendation(track=track, relevance=row["relevance"], signals=row["signals"])
        )
    return out


def embed_query(query: str) -> np.ndarray:
    """CLAP-encode a text query, memoized in Redis.

    Normalizing case and whitespace first — "Rainy Day " and "rainy day" are
    the same query and should not each pay for a forward pass.
    """
    normalized = " ".join(query.lower().split())
    key = cache.make_key("clap-text", q=normalized)

    cached = cache.get_vector(key)
    if cached is not None and cached.size:
        return cached

    vector = encode_query(normalized)
    cache.set_vector(key, vector)
    return vector


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

    key = _ranking_cache_key(
        seed=seed_track_id, query=None, variant=variant, top_k=top_k,
        tags=tags, tempo_range=tempo_range,
    )
    cached = cache.get_json(key)
    if cached:
        return _hydrate(db, cached)

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
    ranked = _rank(candidates, _track_payload(seed), weights, top_k)
    cache.set_json(key, ranked)
    return _hydrate(db, ranked)


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

    Raises `TextSearchUnavailable` if the deployment has no text encoder.
    """
    key = _ranking_cache_key(
        seed=None, query=query.lower().strip(), variant=variant, top_k=top_k,
        tags=tags, tempo_range=tempo_range,
    )
    cached = cache.get_json(key)
    if cached:
        return _hydrate(db, cached)

    weights = get_variant(variant)
    query_vector = embed_query(query)
    candidates = store.search(
        query_vector,
        limit=CANDIDATE_POOL,
        tags=tags,
        tempo_range=tempo_range,
        with_vectors=weights.mmr_lambda < 1.0,
    )
    # No seed track: descriptor and tag similarity have nothing to compare
    # against, so relevance falls back to the audio-text alignment.
    ranked = _rank(candidates, {}, weights, top_k)
    cache.set_json(key, ranked)
    return _hydrate(db, ranked)


def _rank(
    candidates: list[dict[str, Any]],
    seed_payload: dict[str, Any],
    weights: RankingWeights,
    top_k: int,
) -> list[dict[str, Any]]:
    """Score, diversify, and reduce to a cacheable [{track_id, relevance, signals}]."""
    if not candidates:
        return []

    scored = score_candidates(seed_payload, candidates, weights)

    # Vectors ride along from the ANN query when the variant diversifies, so
    # MMR costs no extra requests.
    vectors: dict[str, np.ndarray] = {
        cand["track_id"]: cand["_vector"] for cand in scored if "_vector" in cand
    }

    selected = mmr_select(scored, vectors, top_k=top_k, mmr_lambda=weights.mmr_lambda)
    return [
        {
            "track_id": cand["track_id"],
            "relevance": cand["relevance"],
            "signals": cand["signals"],
        }
        for cand in selected
    ]
