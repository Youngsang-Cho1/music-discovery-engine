"""Descriptor tests on synthetic signals with known ground truth.

Real audio would make these assertions unfalsifiable — a sine at a known
frequency and a click track at a known BPM will not.
"""
import numpy as np
import pytest

from app.embeddings.descriptors import (
    DESCRIPTOR_KEYS,
    compute_descriptors,
    descriptor_vector,
    normalize_descriptors,
)

SR = 22_050


def _sine(freq: float, seconds: float = 5.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _click_track(bpm: float, seconds: float = 10.0) -> np.ndarray:
    """Impulses at a fixed BPM — unambiguous input for beat tracking."""
    signal = np.zeros(int(SR * seconds), dtype=np.float32)
    step = int(SR * 60.0 / bpm)
    for i in range(0, len(signal) - 64, step):
        # Short decaying burst so onset detection has an edge to find.
        burst = np.hanning(64).astype(np.float32)
        signal[i : i + 64] += burst
    return signal


def test_empty_waveform_returns_nothing():
    assert compute_descriptors(np.array([], dtype=np.float32), SR) == {}


def test_all_descriptor_keys_are_produced():
    out = compute_descriptors(_sine(440.0), SR)
    assert set(out) == set(DESCRIPTOR_KEYS)
    assert all(np.isfinite(v) for v in out.values())


def test_brightness_tracks_pitch():
    """Spectral centroid must be higher for a higher tone."""
    low = compute_descriptors(_sine(220.0), SR)["brightness"]
    high = compute_descriptors(_sine(4000.0), SR)["brightness"]
    assert high > low * 2


def test_loudness_tracks_amplitude():
    quiet = compute_descriptors(_sine(440.0, amplitude=0.01), SR)["loudness_db"]
    loud = compute_descriptors(_sine(440.0, amplitude=0.9), SR)["loudness_db"]
    assert loud > quiet
    assert loud <= 0.0  # dBFS is never positive for a signal within [-1, 1]


def test_digital_silence_does_not_produce_negative_infinity():
    """log(0) would poison every downstream distance computation."""
    out = compute_descriptors(np.zeros(SR * 2, dtype=np.float32), SR)
    assert np.isfinite(out["loudness_db"])


def test_tempo_estimate_is_in_the_right_neighbourhood():
    out = compute_descriptors(_click_track(120.0), SR)
    # Beat trackers legitimately land on half/double time; accept the octave.
    assert out["tempo_bpm"] == pytest.approx(120.0, rel=0.1) or out[
        "tempo_bpm"
    ] == pytest.approx(60.0, rel=0.1)


def test_percussive_signal_scores_higher_than_a_pure_tone():
    tone = compute_descriptors(_sine(440.0, seconds=10.0), SR)["percussiveness"]
    clicks = compute_descriptors(_click_track(120.0), SR)["percussiveness"]
    assert clicks > tone


def test_onset_density_rises_with_tempo():
    slow = compute_descriptors(_click_track(60.0), SR)["onset_density"]
    fast = compute_descriptors(_click_track(180.0), SR)["onset_density"]
    assert fast > slow


# --- normalization -----------------------------------------------------


def test_normalization_clamps_to_unit_interval():
    out = normalize_descriptors({"tempo_bpm": 10_000.0, "loudness_db": -999.0})
    assert out["tempo_bpm"] == 1.0
    assert out["loudness_db"] == 0.0


def test_normalization_is_monotonic():
    a = normalize_descriptors({"tempo_bpm": 80.0})["tempo_bpm"]
    b = normalize_descriptors({"tempo_bpm": 160.0})["tempo_bpm"]
    assert b > a


def test_descriptor_vector_has_stable_ordering():
    """Ordering must not depend on dict insertion order, or vectors built at
    ingest time won't line up with vectors built at query time."""
    forward = descriptor_vector({"tempo_bpm": 120.0, "loudness_db": -10.0})
    reversed_ = descriptor_vector({"loudness_db": -10.0, "tempo_bpm": 120.0})
    assert np.array_equal(forward, reversed_)
    assert forward.shape == (len(DESCRIPTOR_KEYS),)


def test_descriptor_vector_defaults_missing_keys_to_zero():
    vec = descriptor_vector({"tempo_bpm": 120.0})
    assert vec.shape == (len(DESCRIPTOR_KEYS),)
    assert np.isfinite(vec).all()
