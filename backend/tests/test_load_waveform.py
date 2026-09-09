"""Decoding tests, driven by real container headers rather than raw arrays.

The original implementation trusted `librosa.get_duration` to place the middle
window. That is correct for a whole file and silently wrong for the byte range
the ingestion pipeline actually fetches: the mp3 header still advertises the
full track, so the offset lands past the end of the data and `load` returns an
empty array. Nothing raised — the empty array embedded into a meaningless
vector and went straight into the index.

These tests encode real audio to a container so the header is a genuine header,
then truncate the bytes the way a ranged HTTP request does.
"""
import io

import numpy as np
import pytest
import soundfile as sf

from app.embeddings.clap import load_waveform

SR = 48_000


def _tone(seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _encode(waveform: np.ndarray, fmt: str = "OGG") -> bytes:
    """Encode to a real container so the decoded header is a real header."""
    buffer = io.BytesIO()
    sf.write(buffer, waveform, SR, format=fmt)
    return buffer.getvalue()


def test_returns_the_requested_window_from_a_long_track():
    audio = _encode(_tone(40.0))
    waveform = load_waveform(audio, seconds=10)
    assert waveform.size == pytest.approx(10 * SR, rel=0.02)


def test_window_is_taken_from_the_middle_not_the_head():
    """Intros are often silence; the head is the least representative part."""
    silence = np.zeros(int(SR * 20), dtype=np.float32)
    audio = _encode(np.concatenate([silence, _tone(20.0)]))

    waveform = load_waveform(audio, seconds=10)
    assert np.abs(waveform).max() > 0.1, "middle window should not be the silent intro"


def test_short_track_returns_everything_it_has():
    audio = _encode(_tone(4.0))
    waveform = load_waveform(audio, seconds=10)
    assert waveform.size == pytest.approx(4 * SR, rel=0.02)


def test_truncated_download_still_yields_audio():
    """The regression: a ranged fetch keeps the full-length header.

    Cutting the byte stream in half must yield roughly half the audio, not an
    empty array positioned past the end of what was actually downloaded.
    """
    audio = _encode(_tone(60.0))
    truncated = audio[: len(audio) // 2]

    waveform = load_waveform(truncated, seconds=10)

    assert waveform.size > 0, "truncated download decoded to nothing"
    assert waveform.size == pytest.approx(10 * SR, rel=0.02)
    assert np.abs(waveform).max() > 0.1, "decoded window is silent"


def test_heavily_truncated_download_returns_what_survived():
    """Well past the point where the header's midpoint is unreachable."""
    audio = _encode(_tone(60.0))
    truncated = audio[: len(audio) // 8]

    waveform = load_waveform(truncated, seconds=10)

    assert waveform.size > 0
    assert np.isfinite(waveform).all()
