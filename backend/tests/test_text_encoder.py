import numpy as np
import pytest

from app.config import Settings, get_settings
from app.embeddings import text_encoder
from app.embeddings.text_encoder import TextSearchUnavailable, encode_query


@pytest.fixture
def settings_with(monkeypatch):
    """Override settings without touching the process environment."""

    def _apply(**overrides):
        get_settings.cache_clear()
        base = Settings(**overrides)
        monkeypatch.setattr(text_encoder, "get_settings", lambda: base)
        return base

    yield _apply
    get_settings.cache_clear()


def test_disabled_mode_raises_a_typed_error(settings_with):
    settings_with(text_encoder_mode="disabled")
    with pytest.raises(TextSearchUnavailable, match="disabled"):
        encode_query("rainy day piano")


def test_remote_mode_without_a_url_is_a_config_error(settings_with):
    settings_with(text_encoder_mode="remote", text_encoder_url="")
    with pytest.raises(TextSearchUnavailable, match="not configured"):
        encode_query("rainy day piano")


def test_remote_mode_returns_the_service_vector(settings_with, monkeypatch):
    settings_with(text_encoder_mode="remote", text_encoder_url="http://embedder")
    expected = np.arange(512, dtype=np.float32)

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"embedding": expected.tolist()}

    monkeypatch.setattr(text_encoder.httpx, "post", lambda *a, **k: Response())
    assert np.allclose(encode_query("x"), expected)


def test_remote_failure_is_reported_as_unavailable(settings_with, monkeypatch):
    """A dead embedder is a dependency outage the route turns into a 503 —
    it must not surface as an unhandled 500."""
    settings_with(text_encoder_mode="remote", text_encoder_url="http://embedder")

    def boom(*_args, **_kwargs):
        raise ConnectionError("refused")

    monkeypatch.setattr(text_encoder.httpx, "post", boom)
    with pytest.raises(TextSearchUnavailable, match="unreachable"):
        encode_query("x")


def test_wrong_dimensionality_is_rejected(settings_with, monkeypatch):
    """A mismatched vector would be silently wrong in the ANN index rather
    than loudly wrong here."""
    settings_with(text_encoder_mode="remote", text_encoder_url="http://embedder")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"embedding": [0.1, 0.2, 0.3]}

    monkeypatch.setattr(text_encoder.httpx, "post", lambda *a, **k: Response())
    with pytest.raises(TextSearchUnavailable, match="shape"):
        encode_query("x")
