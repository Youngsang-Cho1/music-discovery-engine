"""Shared FastAPI dependencies."""
from functools import lru_cache

from app.retrieval.vector_store import VectorStore


@lru_cache(maxsize=1)
def _store() -> VectorStore:
    return VectorStore()


def get_store() -> VectorStore:
    """Process-wide Qdrant client. The client is thread-safe and holds a
    connection pool, so one instance per process is what we want."""
    return _store()
