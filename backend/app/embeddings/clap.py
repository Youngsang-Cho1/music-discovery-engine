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
from functools import lru_cache

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    """Lazily load CLAP. Import torch/transformers here so that API processes
    that never embed anything don't pay the import cost."""
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
    return model, processor, device


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


def load_waveform(audio_bytes: bytes, *, seconds: int | None = None) -> np.ndarray:
    """Decode arbitrary audio bytes to a mono waveform at CLAP's sample rate.

    Reads the middle window rather than the head: intros are frequently silence
    or a fade-in and are the least representative part of a track.
    """
    import librosa

    settings = get_settings()
    seconds = seconds or settings.clap_window_seconds

    duration = librosa.get_duration(path=io.BytesIO(audio_bytes))
    offset = max(0.0, (duration - seconds) / 2)

    waveform, _ = librosa.load(
        io.BytesIO(audio_bytes),
        sr=settings.clap_sample_rate,
        mono=True,
        offset=offset,
        duration=seconds,
    )
    return waveform


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
