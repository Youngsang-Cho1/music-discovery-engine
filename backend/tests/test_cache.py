import numpy as np
import pytest

from app import cache


@pytest.fixture(autouse=True)
def _no_redis(monkeypatch):
    """Default to "Redis is down" so tests never touch a real server."""
    monkeypatch.setattr(cache, "_client", lambda: None)


class FakeRedis:
    """Minimal in-memory stand-in for the calls the cache module makes."""

    def __init__(self, fail: bool = False):
        self.store: dict[str, bytes] = {}
        self.fail = fail

    def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return self.store.get(key)

    def setex(self, key, _ttl, value):
        if self.fail:
            raise ConnectionError("redis down")
        self.store[key] = value if isinstance(value, bytes) else value.encode()

    def scan_iter(self, match, count=500):  # noqa: ARG002 - signature parity
        prefix = match.rstrip("*")
        return [k for k in list(self.store) if k.startswith(prefix)]

    def delete(self, key):
        return 1 if self.store.pop(key, None) is not None else 0


# --- key construction --------------------------------------------------


def test_key_is_order_independent():
    """Filter dicts arrive in arbitrary order; one logical query must map to
    one key or the cache silently fragments."""
    a = cache.make_key("rank", variant="control", top_k=10, seed="t1")
    b = cache.make_key("rank", seed="t1", top_k=10, variant="control")
    assert a == b


def test_key_changes_with_any_input():
    base = cache.make_key("rank", seed="t1", variant="control")
    assert base != cache.make_key("rank", seed="t2", variant="control")
    assert base != cache.make_key("rank", seed="t1", variant="diverse")


def test_key_namespaces_are_separate():
    assert cache.make_key("rank", q="x") != cache.make_key("clap-text", q="x")


def test_key_is_namespaced_for_scan():
    assert cache.make_key("rank", q="x").startswith("mde:rank:")


# --- degradation without Redis ----------------------------------------


def test_reads_return_none_without_redis():
    assert cache.get_vector("mde:clap-text:abc") is None
    assert cache.get_json("mde:rank:abc") is None


def test_writes_are_silent_without_redis():
    """A cache write must never raise into the request path."""
    cache.set_vector("mde:clap-text:abc", np.ones(4, dtype=np.float32))
    cache.set_json("mde:rank:abc", [{"track_id": "t1"}])


def test_invalidate_is_zero_without_redis():
    assert cache.invalidate("rank") == 0


# --- round trips -------------------------------------------------------


def test_vector_round_trip_preserves_values(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(cache, "_client", lambda: fake)

    vector = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    cache.set_vector("mde:clap-text:k", vector)
    assert np.allclose(cache.get_vector("mde:clap-text:k"), vector)


def test_json_round_trip(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(cache, "_client", lambda: fake)

    ranked = [{"track_id": "t1", "relevance": 0.9, "signals": {"audio": 0.9}}]
    cache.set_json("mde:rank:k", ranked)
    assert cache.get_json("mde:rank:k") == ranked


def test_a_failing_redis_degrades_to_a_miss(monkeypatch):
    """Redis erroring mid-request must look like a cache miss, not a 500."""
    monkeypatch.setattr(cache, "_client", lambda: FakeRedis(fail=True))

    cache.set_json("mde:rank:k", [{"track_id": "t1"}])  # must not raise
    assert cache.get_json("mde:rank:k") is None
    assert cache.get_vector("mde:clap-text:k") is None


def test_invalidate_only_clears_its_namespace(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(cache, "_client", lambda: fake)

    cache.set_json(cache.make_key("rank", q="a"), [1])
    cache.set_json(cache.make_key("rank", q="b"), [2])
    text_key = cache.make_key("clap-text", q="a")
    cache.set_vector(text_key, np.ones(3, dtype=np.float32))

    removed = cache.invalidate("rank")

    assert removed == 2
    # Ingesting new tracks invalidates rankings, but text embeddings stay
    # valid — the model didn't change, only the corpus.
    assert cache.get_vector(text_key) is not None
