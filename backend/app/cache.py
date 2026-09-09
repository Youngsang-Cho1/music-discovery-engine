"""Redis cache.

Two things in the request path are expensive and highly repeatable:

* **CLAP text encoding** — a transformer forward pass per query. Query
  distributions are heavily skewed (the example prompts on the landing page
  alone will dominate), so the same vector gets recomputed constantly.
* **Recommendation responses** — same seed, same arm, same filters produces the
  same ranking, and popular seeds are requested repeatedly.

Every failure path here degrades to a cache miss rather than an error: a cache
that can take the API down with it is worse than no cache.
"""
import hashlib
import json
import logging
from functools import lru_cache
from typing import Any

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

TEXT_EMBEDDING_TTL = 24 * 60 * 60  # queries are stable; the model never changes
RECOMMENDATION_TTL = 5 * 60  # short: the catalog grows under us


@lru_cache(maxsize=1)
def _client():
    """Redis client, or None if Redis isn't reachable."""
    try:
        import redis

        client = redis.Redis.from_url(
            get_settings().redis_url,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        client.ping()
        return client
    except Exception as exc:
        logger.warning("redis unavailable, running without cache: %s", exc)
        return None


def make_key(namespace: str, **parts: Any) -> str:
    """Deterministic key from keyword parts.

    Sorted and hashed so that filter dicts with different insertion orders
    don't produce two entries for one logical query.
    """
    payload = json.dumps(parts, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:32]
    return f"mde:{namespace}:{digest}"


def get_vector(key: str) -> np.ndarray | None:
    client = _client()
    if client is None:
        return None
    try:
        raw = client.get(key)
    except Exception:
        return None
    if raw is None:
        return None
    return np.frombuffer(raw, dtype=np.float32)


def set_vector(key: str, vector: np.ndarray, ttl: int = TEXT_EMBEDDING_TTL) -> None:
    client = _client()
    if client is None:
        return
    try:
        client.setex(key, ttl, vector.astype(np.float32).tobytes())
    except Exception:
        pass


def get_json(key: str) -> Any | None:
    client = _client()
    if client is None:
        return None
    try:
        raw = client.get(key)
    except Exception:
        return None
    return json.loads(raw) if raw else None


def set_json(key: str, value: Any, ttl: int = RECOMMENDATION_TTL) -> None:
    client = _client()
    if client is None:
        return
    try:
        client.setex(key, ttl, json.dumps(value, default=str))
    except Exception:
        pass


def invalidate(namespace: str) -> int:
    """Drop every key in a namespace. Called after ingestion adds tracks.

    Uses SCAN rather than KEYS — KEYS blocks the whole server while it walks
    the keyspace.
    """
    client = _client()
    if client is None:
        return 0
    removed = 0
    try:
        for key in client.scan_iter(match=f"mde:{namespace}:*", count=500):
            removed += client.delete(key)
    except Exception:
        pass
    return removed
