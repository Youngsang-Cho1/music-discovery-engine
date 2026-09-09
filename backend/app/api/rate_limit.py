"""Redis-backed sliding-window rate limiting.

A fixed-window counter is simpler but lets a caller send a full window's budget
at the end of one window and again at the start of the next — double the
intended rate across the boundary. A sorted set keyed by timestamp costs one
extra command and doesn't have that edge.

Budgets differ per route because the costs differ by orders of magnitude: a
seed-track recommendation is two Qdrant calls, while an uncached text query is
a transformer forward pass.

Fails open. Redis being down should slow nobody down; the alternative is a
cache outage turning into a total outage.
"""
import logging
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

from app import cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Budget:
    """`limit` requests per `window_seconds`."""

    name: str
    limit: int
    window_seconds: int


# Generous enough that a human clicking around never notices; tight enough that
# a script can't drain the Jamendo quota or pin the embedder.
READ_BUDGET = Budget("read", limit=120, window_seconds=60)
SEARCH_BUDGET = Budget("search", limit=20, window_seconds=60)
EVENT_BUDGET = Budget("event", limit=600, window_seconds=60)


def _client_key(request: Request, budget: Budget) -> str:
    # X-Forwarded-For's first hop is the client when running behind a single
    # trusted proxy, which is what Railway/Vercel put in front of us.
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (
        request.client.host if request.client else "unknown"
    )
    return f"mde:rl:{budget.name}:{ip}"


def _consume(key: str, budget: Budget) -> tuple[bool, int]:
    """Record one hit. Returns (allowed, remaining)."""
    client = cache.get_client()
    if client is None:
        return True, budget.limit  # fail open

    now = time.time()
    cutoff = now - budget.window_seconds

    try:
        pipe = client.pipeline()
        pipe.zremrangebyscore(key, 0, cutoff)  # drop hits that aged out
        pipe.zcard(key)
        pipe.zadd(key, {f"{now}:{id(pipe)}": now})
        pipe.expire(key, budget.window_seconds + 1)
        _, count, _, _ = pipe.execute()
    except Exception as exc:
        logger.warning("rate limit check failed, allowing: %s", exc)
        return True, budget.limit

    # `count` was read before this request was added.
    return count < budget.limit, max(0, budget.limit - count - 1)


def rate_limit(budget: Budget):
    """FastAPI dependency enforcing `budget`."""

    def dependency(request: Request) -> None:
        allowed, remaining = _consume(_client_key(request, budget), budget)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded: {budget.limit} requests per "
                    f"{budget.window_seconds}s. Try again shortly."
                ),
                headers={"Retry-After": str(budget.window_seconds)},
            )
        request.state.rate_limit_remaining = remaining

    return dependency
