"""Audio-derived scalar descriptors.

v1 took `danceability`, `energy`, `valence` etc. straight from a Spotify export.
That is a proprietary dependency the deployed service can't rely on, and it is
undefined for any track outside Spotify's catalog — including every track we
now ingest from Jamendo.

These descriptors are computed from the waveform instead, so any audio we can
decode gets them. They are deliberately interpretable (tempo in BPM, loudness in
dBFS) because they double as user-facing filter controls in the UI, while CLAP
carries the actual semantic similarity.
"""
import numpy as np

# Normalization ranges used to map raw units into [0, 1] so the reranker sees
# comparable scales. Bounds are conventional audio ranges, not learned.
_RANGES = {
    "tempo_bpm": (40.0, 220.0),
    "loudness_db": (-60.0, 0.0),
    "brightness": (0.0, 8000.0),
    "percussiveness": (0.0, 1.0),
    "dynamic_range": (0.0, 0.3),
    "onset_density": (0.0, 8.0),
}


def compute_descriptors(waveform: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Compute interpretable scalar descriptors for one waveform."""
    import librosa

    if waveform.size == 0:
        return {}

    tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate)
    tempo_bpm = float(np.atleast_1d(tempo)[0])

    rms = librosa.feature.rms(y=waveform)[0]
    # dBFS relative to full scale; guards against log(0) on digital silence.
    loudness_db = float(20 * np.log10(max(float(np.mean(rms)), 1e-10)))

    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)[0]

    # Harmonic/percussive source separation: the percussive share of total
    # energy is a reasonable stand-in for "beat-driven vs. textural".
    harmonic, percussive = librosa.effects.hpss(waveform)
    h_energy = float(np.sum(harmonic**2))
    p_energy = float(np.sum(percussive**2))
    percussiveness = p_energy / (h_energy + p_energy + 1e-10)

    onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sample_rate)
    duration_sec = len(waveform) / sample_rate
    onset_density = len(onsets) / max(duration_sec, 1e-6)

    return {
        "tempo_bpm": tempo_bpm,
        "loudness_db": loudness_db,
        "brightness": float(np.mean(centroid)),
        "percussiveness": float(percussiveness),
        "dynamic_range": float(np.std(rms)),
        "onset_density": float(onset_density),
    }


def normalize_descriptors(descriptors: dict[str, float]) -> dict[str, float]:
    """Map raw descriptors into [0, 1] using fixed conventional ranges."""
    out: dict[str, float] = {}
    for key, value in descriptors.items():
        low, high = _RANGES.get(key, (0.0, 1.0))
        out[key] = float(np.clip((value - low) / (high - low), 0.0, 1.0))
    return out


def descriptor_vector(descriptors: dict[str, float]) -> np.ndarray:
    """Stable-ordered normalized vector, for distance computations."""
    normalized = normalize_descriptors(descriptors)
    return np.array([normalized.get(k, 0.0) for k in sorted(_RANGES)], dtype=np.float32)


DESCRIPTOR_KEYS = sorted(_RANGES)
