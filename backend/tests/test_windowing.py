"""Window selection and pooling.

A single 10s window is a high-variance sample of a track — measured on real
audio, two windows of the same song can score ~0.0 cosine against each other.
The pooling that fixes it only works if the windows actually span the track,
so the offset arithmetic is worth pinning down.
"""
import io

import numpy as np
import pytest
import soundfile as sf

from app.embeddings.clap import decode, load_windows, window_offsets

SR = 48_000
WINDOW = 10 * SR


def _tone(seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _encode(waveform: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, waveform, SR, format="OGG")
    return buffer.getvalue()


# --- offsets -----------------------------------------------------------


def test_offsets_span_the_whole_track():
    """First window starts at 0, last one ends at the final sample — otherwise
    the tail of every track is never looked at."""
    total, count = 100_000, 3
    offsets = window_offsets(total, window=10_000, count=count)

    assert offsets[0] == 0
    assert offsets[-1] + 10_000 == total
    assert len(offsets) == count


def test_offsets_are_evenly_spaced():
    offsets = window_offsets(100_000, window=10_000, count=4)
    # Pairwise over consecutive offsets — the second list is one shorter by
    # construction, so strict zip would reject it.
    gaps = [b - a for a, b in zip(offsets, offsets[1:], strict=False)]
    assert len(set(gaps)) == 1


def test_offsets_are_ascending_and_in_range():
    offsets = window_offsets(97_531, window=10_000, count=5)
    assert offsets == sorted(offsets)
    assert all(0 <= o <= 97_531 - 10_000 for o in offsets)


def test_single_window_is_taken_from_the_middle():
    """With one window there is no spanning to do, so take the most
    representative slice rather than the intro."""
    assert window_offsets(100_000, window=10_000, count=1) == [45_000]


def test_track_shorter_than_the_window_starts_at_zero():
    assert window_offsets(5_000, window=10_000, count=3) == [0]


def test_offsets_never_overrun_the_buffer():
    """Slicing past the end would silently yield a short array, which embeds
    into a vector describing a fragment."""
    total, window = 30_001, 10_000
    for count in (1, 2, 3, 7):
        for offset in window_offsets(total, window, count):
            assert offset + window <= total


# --- decoding ----------------------------------------------------------


def test_load_windows_returns_the_requested_count():
    windows = load_windows(_encode(_tone(60.0)), count=3)
    assert len(windows) == 3
    assert all(w.size == WINDOW for w in windows)


def test_load_windows_covers_different_audio():
    """Three windows of a track whose sections differ must not be identical —
    otherwise pooling averages three copies of one thing."""
    sections = np.concatenate([_tone(20.0, 200.0), _tone(20.0, 900.0), _tone(20.0, 3500.0)])
    windows = load_windows(_encode(sections), count=3)

    assert len(windows) == 3
    assert not np.allclose(windows[0], windows[-1])


def test_short_track_yields_one_window():
    windows = load_windows(_encode(_tone(4.0)), count=3)
    assert len(windows) == 1
    assert windows[0].size == pytest.approx(4 * SR, rel=0.02)


def test_silence_still_decodes_rather_than_vanishing():
    windows = load_windows(_encode(np.zeros(int(SR * 30), dtype=np.float32)), count=3)
    assert len(windows) == 3
    assert all(np.isfinite(w).all() for w in windows)


def test_decode_reads_what_is_present_not_what_the_header_claims():
    """A ranged download keeps the full-length header; trusting it produced an
    empty array and a meaningless embedding."""
    audio = _encode(_tone(60.0))
    truncated = audio[: len(audio) // 3]

    decoded = decode(truncated)
    assert decoded.size > 0
    assert decoded.size < 60 * SR


def test_windows_from_a_truncated_download_are_still_usable():
    audio = _encode(_tone(60.0))
    windows = load_windows(audio[: len(audio) // 3], count=3)

    assert windows
    assert all(w.size > 0 for w in windows)
    assert all(np.abs(w).max() > 0.1 for w in windows), "decoded windows are silent"
