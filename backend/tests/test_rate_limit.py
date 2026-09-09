import time

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app import cache
from app.api.rate_limit import Budget, rate_limit


class FakeRedis:
    """Enough of a Redis sorted set for the sliding window."""

    def __init__(self, fail: bool = False):
        self.sets: dict[str, dict[str, float]] = {}
        self.fail = fail
        self.ops: list[str] = []

    def pipeline(self):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis: FakeRedis):
        self.redis = redis
        self.queued = []

    def zremrangebyscore(self, key, lo, hi):
        self.queued.append(("zrem", key, lo, hi))
        return self

    def zcard(self, key):
        self.queued.append(("zcard", key))
        return self

    def zadd(self, key, mapping):
        self.queued.append(("zadd", key, mapping))
        return self

    def expire(self, key, ttl):
        self.queued.append(("expire", key, ttl))
        return self

    def execute(self):
        if self.redis.fail:
            raise ConnectionError("redis down")
        results = []
        for op in self.queued:
            kind, key = op[0], op[1]
            self.redis.ops.append(kind)
            bucket = self.redis.sets.setdefault(key, {})
            if kind == "zrem":
                _, _, lo, hi = op
                for member in [m for m, s in bucket.items() if lo <= s <= hi]:
                    del bucket[member]
                results.append(None)
            elif kind == "zcard":
                results.append(len(bucket))
            elif kind == "zadd":
                bucket.update(op[2])
                results.append(1)
            else:
                results.append(True)
        return results


BUDGET = Budget("test", limit=3, window_seconds=60)


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/limited", dependencies=[Depends(rate_limit(BUDGET))])
    def limited():
        return {"ok": True}

    return app


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(cache, "_client", lambda: fake)
    return fake


def test_requests_within_budget_pass(redis):
    client = TestClient(_app())
    for _ in range(BUDGET.limit):
        assert client.get("/limited").status_code == 200


def test_request_over_budget_is_429(redis):
    client = TestClient(_app())
    for _ in range(BUDGET.limit):
        client.get("/limited")

    resp = client.get("/limited")
    assert resp.status_code == 429
    assert "Rate limit exceeded" in resp.json()["detail"]


def test_429_tells_the_caller_when_to_retry(redis):
    client = TestClient(_app())
    for _ in range(BUDGET.limit + 1):
        resp = client.get("/limited")
    assert resp.headers["retry-after"] == str(BUDGET.window_seconds)


def test_old_hits_age_out_of_the_window(redis, monkeypatch):
    """The window slides: hits older than window_seconds stop counting."""
    client = TestClient(_app())
    for _ in range(BUDGET.limit):
        client.get("/limited")
    assert client.get("/limited").status_code == 429

    # Jump past the window without sleeping through it.
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + BUDGET.window_seconds + 1)

    assert client.get("/limited").status_code == 200


def test_budgets_are_tracked_separately_per_client(redis):
    client = TestClient(_app())
    for _ in range(BUDGET.limit):
        client.get("/limited", headers={"X-Forwarded-For": "1.1.1.1"})

    assert client.get("/limited", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429
    # A different caller must not inherit the first one's exhausted budget.
    assert client.get("/limited", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200


def test_forwarded_for_uses_the_first_hop(redis):
    """Behind a proxy the client is the leftmost entry; using the last would
    bucket every user under the proxy's own address."""
    client = TestClient(_app())
    for _ in range(BUDGET.limit):
        client.get("/limited", headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.1"})

    assert (
        client.get("/limited", headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.2"}).status_code
        == 429
    )


def test_redis_down_fails_open(monkeypatch):
    """A cache outage must not become an API outage."""
    monkeypatch.setattr(cache, "_client", lambda: None)
    client = TestClient(_app())
    for _ in range(BUDGET.limit * 3):
        assert client.get("/limited").status_code == 200


def test_redis_erroring_mid_request_fails_open(monkeypatch):
    monkeypatch.setattr(cache, "_client", lambda: FakeRedis(fail=True))
    client = TestClient(_app())
    for _ in range(BUDGET.limit * 3):
        assert client.get("/limited").status_code == 200
