"""CLAP audio/text encoder.

v1 described a track with 56 hand-picked librosa statistics (MFCC means/stds,
chroma, spectral contrast). That vector is only meaningful relative to other
vectors built the same way — there is no shared space with language, so text
query was impossible and the metadata modality had to be bolted on with a
hand-tuned lambda.

CLAP embeds audio and text into one contrastively-trained space, which buys two
things at once: better acoustic representations than hand-crafted statistics,
and free natural-language search ("rainy day lo-fi") against the same index.
"""
import io
import logging
import threading

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)


_model_bundle: tuple | None = None
_model_lock = threading.Lock()


def _load_model():
    """Lazily load CLAP, exactly once per process.

    `lru_cache` is not sufficient here. It memoizes the *result* but holds no
    lock while the function runs, so concurrent callers all miss and all start
    loading. FastAPI runs sync handlers in a threadpool, so a burst of cold
    text-search requests each allocated its own ~1.5GB model and the process
    was OOM-killed — found by the load test, not by any unit test.

    Double-checked locking: the fast path stays lock-free once loaded.
    """
    global _model_bundle

    if _model_bundle is not None:
        return _model_bundle

    with _model_lock:
        # Re-check: another thread may have loaded it while we waited.
        if _model_bundle is not None:
            return _model_bundle

        # Imported here so API processes that never embed skip the cost.
        import torch
        from transformers import ClapModel, ClapProcessor

        settings = get_settings()
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info("loading CLAP %s on %s", settings.clap_model_id, device)
        model = ClapModel.from_pretrained(settings.clap_model_id).to(device).eval()
        processor = ClapProcessor.from_pretrained(settings.clap_model_id)
        _model_bundle = (model, processor, device)

    return _model_bundle


def warm_up() -> None:
    """Load the model ahead of serving traffic.

    Without this the first request pays a ~13s load, and everything queued
    behind it waits on the same lock.
    """
    _load_model()


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norm == 0, 1.0, norm)


def _as_vectors(output) -> np.ndarray:
    """Extract the projected embedding from a CLAP feature call.

    transformers <5 returned a plain tensor from `get_*_features`; 5.x returns
    a `BaseModelOutputWithPooling` whose `pooler_output` holds the projected
    vector. Accept both so the pinned version isn't load-bearing.
    """
    tensor = output if hasattr(output, "cpu") else output.pooler_output
    return _l2_normalize(tensor.cpu().numpy().astype(np.float32))


def decode(audio_bytes: bytes) -> np.ndarray:
    """Decode audio bytes to a full mono waveform at CLAP's sample rate.

    Decodes whatever is actually present rather than trusting the container
    header. Ingestion fetches a byte range, so an mp3's header still advertises
    the full track length; seeking by it lands past the end of the data we hold
    and yields an empty array, which then embeds into a meaningless vector
    instead of failing.
    """
    import librosa

    waveform, _ = librosa.load(
        io.BytesIO(audio_bytes), sr=get_settings().clap_sample_rate, mono=True
    )
    return waveform


def window_offsets(total_samples: int, window: int, count: int) -> list[int]:
    """Evenly spaced window start offsets covering the decoded audio."""
    if total_samples <= window:
        return [0]
    if count <= 1:
        return [(total_samples - window) // 2]
    span = total_samples - window
    return [int(round(i * span / (count - 1))) for i in range(count)]


def load_windows(
    audio_bytes: bytes, *, count: int | None = None, seconds: int | None = None
) -> list[np.ndarray]:
    """Decode and slice a track into the windows that will be embedded.

    One window is not a description of a song. Measured across real tracks,
    two 10s windows of the *same* track score as low as ~0.0 cosine against
    each other — the same magnitude as the gap between unrelated tracks. Which
    section happens to be sampled would then decide what the track "sounds
    like", so several windows are taken and pooled instead.
    """
    settings = get_settings()
    seconds = seconds or settings.clap_window_seconds
    count = count or settings.clap_windows_per_track

    waveform = decode(audio_bytes)
    window = int(seconds * settings.clap_sample_rate)

    if waveform.size == 0:
        return []
    if waveform.size <= window:
        return [waveform]

    return [waveform[o : o + window] for o in window_offsets(waveform.size, window, count)]


def load_waveform(audio_bytes: bytes, *, seconds: int | None = None) -> np.ndarray:
    """Single middle window. Retained for callers that want one representative
    slice (descriptor extraction) rather than the pooled set."""
    settings = get_settings()
    seconds = seconds or settings.clap_window_seconds

    waveform = decode(audio_bytes)
    window = int(seconds * settings.clap_sample_rate)
    if waveform.size <= window:
        return waveform

    start = (waveform.size - window) // 2
    return waveform[start : start + window]


def embed_track(windows: list[np.ndarray]) -> np.ndarray:
    """Mean-pool per-window embeddings into one L2-normalized track vector.

    Mean rather than max: the windows are alternative views of one track, and
    averaging unit vectors moves the result toward whatever they agree on while
    the parts they disagree about cancel. Max would just pick one window back
    out and reintroduce the variance this exists to remove.
    """
    if not windows:
        raise ValueError("no windows to embed")

    per_window = embed_audio(windows)
    pooled = per_window.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm == 0:
        # Windows cancelled exactly; fall back to a single one rather than
        # emitting a zero vector that would match everything equally.
        return per_window[0]
    return (pooled / norm).astype(np.float32)


def embed_audio(waveforms: list[np.ndarray]) -> np.ndarray:
    """Embed a batch of mono waveforms. Returns L2-normalized (n, dim) floats."""
    import torch

    model, processor, device = _load_model()
    settings = get_settings()

    # `audio=` (singular) — transformers 5 removed the old `audios=` alias.
    inputs = processor(
        audio=waveforms,
        sampling_rate=settings.clap_sample_rate,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        features = model.get_audio_features(**inputs)

    return _as_vectors(features)


def embed_text(queries: list[str]) -> np.ndarray:
    """Embed natural-language queries into the same space as the audio."""
    import torch

    model, processor, device = _load_model()

    inputs = processor(text=queries, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)

    return _as_vectors(features)
