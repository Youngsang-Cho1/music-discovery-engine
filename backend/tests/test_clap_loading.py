"""Regression test for concurrent model loading.

`@lru_cache` memoizes a result but takes no lock while the function runs, so
every thread that arrives before the first load finishes misses the cache and
starts its own load. FastAPI runs sync route handlers in a threadpool, so a
burst of cold text-search requests each allocated a separate ~1.5GB CLAP model
and the process was OOM-killed. The load test found it; nothing else did.

These tests stub the actual model construction — the property under test is
"how many times does construction happen", not what it produces.
"""
import threading
import time

import pytest

from app.embeddings import clap


@pytest.fixture(autouse=True)
def _reset_model_state():
    clap._model_bundle = None
    yield
    clap._model_bundle = None


@pytest.fixture
def slow_loader(monkeypatch):
    """Replace the real load with a slow counter, widening the race window."""
    calls = []

    def fake_load():
        # Double-checked locking, with construction replaced by a sleep.
        if clap._model_bundle is not None:
            return clap._model_bundle
        with clap._model_lock:
            if clap._model_bundle is not None:
                return clap._model_bundle
            calls.append(1)
            time.sleep(0.2)  # stand-in for a multi-second model load
            clap._model_bundle = ("model", "processor", "cpu")
        return clap._model_bundle

    monkeypatch.setattr(clap, "_load_model", fake_load)
    return calls


def test_concurrent_callers_load_the_model_once(slow_loader):
    """32 threads racing a cold cache must produce exactly one load."""
    threads = [threading.Thread(target=clap._load_model) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(slow_loader) == 1, f"model loaded {len(slow_loader)} times"


def test_all_callers_get_the_same_instance(slow_loader):
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(clap._load_model())) for _ in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 16
    assert all(r is results[0] for r in results), "callers got different model instances"


def test_warm_up_populates_the_cache(monkeypatch):
    loads = []

    def fake_load():
        loads.append(1)
        clap._model_bundle = ("model", "processor", "cpu")
        return clap._model_bundle

    monkeypatch.setattr(clap, "_load_model", fake_load)
    monkeypatch.setattr(clap, "warm_up", lambda: fake_load())

    clap.warm_up()
    assert clap._model_bundle is not None
    assert len(loads) == 1


def test_second_call_skips_the_lock_entirely(slow_loader):
    """Once loaded, the fast path must not serialize on the lock."""
    clap._load_model()
    assert len(slow_loader) == 1

    clap._model_lock.acquire()  # hold it: a correct fast path never waits
    try:
        assert clap._load_model() is not None
    finally:
        clap._model_lock.release()
