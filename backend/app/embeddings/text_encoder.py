"""Resolves a text query to a vector, by whichever route deployment allows.

Text search is the only endpoint that needs CLAP resident at request time —
seed-track recommendation reads vectors computed during ingestion. Loading
~2.5GB of torch into the API process just to serve that one route is what
forces an otherwise-small service onto a large instance, so the route is
pluggable instead.
"""
import logging

import httpx
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)


class TextSearchUnavailable(RuntimeError):
    """Raised when text search is switched off or its backend is unreachable."""


def encode_query(query: str) -> np.ndarray:
    """Encode `query` into the shared CLAP space."""
    settings = get_settings()
    mode = settings.text_encoder_mode

    if mode == "disabled":
        raise TextSearchUnavailable(
            "Text search is disabled on this deployment. Seed-track "
            "recommendations are unaffected."
        )

    if mode == "remote":
        return _encode_remote(query, settings.text_encoder_url, settings.text_encoder_timeout)

    from app.embeddings.clap import embed_text

    return embed_text([query])[0]


def _encode_remote(query: str, url: str, timeout: float) -> np.ndarray:
    if not url:
        raise TextSearchUnavailable("TEXT_ENCODER_URL is not configured")

    try:
        resp = httpx.post(url.rstrip("/") + "/embed", json={"query": query}, timeout=timeout)
        resp.raise_for_status()
        vector = np.asarray(resp.json()["embedding"], dtype=np.float32)
    except Exception as exc:
        # Surfaced as a 503 rather than a 500: the embedder being down is a
        # dependency outage, not a bug in this request.
        raise TextSearchUnavailable(f"Embedding service unreachable: {exc}") from exc

    if vector.ndim != 1 or vector.size != get_settings().embedding_dim:
        raise TextSearchUnavailable(
            f"Embedding service returned shape {vector.shape}, "
            f"expected ({get_settings().embedding_dim},)"
        )
    return vector
