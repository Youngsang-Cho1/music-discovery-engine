"""Is CLAP actually better than hand-crafted librosa features for retrieval?

The README claimed it was. That claim was never measured, and it matters: CLAP
is the only reason the deployment needs a 5.92GB image, ~2.5GB of RAM, and a
separately-scaled embedder service. If it does not beat v1's 56-dimension
MFCC/chroma vector at audio similarity, then its entire value is text search,
and that feature has to justify the complexity on its own.

Two proxies for "these tracks belong together", neither perfect, both
independent of the representations being compared:

* **Same-artist pair discrimination** — the metric v1 reported (83.3%). Does a
  pair of tracks by one artist rank closer than a random pair?
* **Tag overlap correlation** — do embedding distances agree with Jaccard
  overlap on Jamendo's genre/instrument tags? Tags are human-assigned and were
  never seen by either representation.

    python -m scripts.compare_representations --tracks 120
"""
import argparse
import asyncio
import io
import logging
from collections import defaultdict

import httpx
import numpy as np

from app.embeddings.clap import embed_track, window_offsets
from app.ingestion.jamendo import JamendoClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare")
logging.getLogger("httpx").setLevel(logging.WARNING)

SR = 48_000
WINDOW = 10 * SR
PREVIEW_BYTES = 6_000_000


def librosa_features(waveform: np.ndarray) -> np.ndarray:
    """v1's 56-dimension vector: MFCC means/stds, chroma, spectral, contrast.

    Reproduced from legacy/src/audio_feature_extractor.py so the comparison is
    against what the project actually shipped, not a strawman.
    """
    import librosa

    y = waveform
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR)
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SR)

    return np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            chroma.mean(axis=1),
            [centroid.mean(), centroid.std()],
            [bandwidth.mean(), bandwidth.std()],
            [rolloff.mean(), rolloff.std()],
            [zcr.mean(), zcr.std()],
            [rms.mean(), rms.std()],
            [float(np.atleast_1d(tempo)[0])],
            contrast.mean(axis=1),
        ]
    ).astype(np.float32)


def _standardize_and_normalize(matrix: np.ndarray) -> np.ndarray:
    """z-score per dimension, then L2 — v1's 'cosine collapse' fix.

    Without this the largest-magnitude dimensions dominate the angle and every
    track looks alike, which would make the comparison unfair to librosa.
    """
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    z = (matrix - mean) / std
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.where(norms == 0, 1.0, norms)


def pair_discrimination(
    vectors: np.ndarray, artist_ids: list[str], rng: np.random.Generator, trials: int = 5000
) -> float:
    """Fraction of trials where a same-artist pair beats a random pair."""
    by_artist = defaultdict(list)
    for i, a in enumerate(artist_ids):
        by_artist[a].append(i)
    multi = [idxs for idxs in by_artist.values() if len(idxs) >= 2]
    if not multi:
        return float("nan")

    wins = 0
    for _ in range(trials):
        group = multi[rng.integers(len(multi))]
        i, j = rng.choice(group, size=2, replace=False)
        k, m = rng.integers(len(vectors), size=2)
        while artist_ids[k] == artist_ids[m]:
            k, m = rng.integers(len(vectors), size=2)
        same = float(vectors[i] @ vectors[j])
        rand = float(vectors[k] @ vectors[m])
        wins += same > rand
    return wins / trials


def tag_correlation(vectors: np.ndarray, tags: list[list[str]]) -> float:
    """Spearman correlation between embedding similarity and tag Jaccard."""
    from scipy import stats

    sims, jaccs = [], []
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = set(tags[i]), set(tags[j])
            if not a or not b:
                continue
            union = a | b
            jaccs.append(len(a & b) / len(union))
            sims.append(float(vectors[i] @ vectors[j]))
    if len(sims) < 10:
        return float("nan")
    return float(stats.spearmanr(sims, jaccs).statistic)


async def collect(target: int):
    jamendo = JamendoClient()
    clap_vecs, lib_vecs, artists, tag_lists, names = [], [], [], [], []

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as http:
        # groupby=artist_id would give one track per artist; we need several.
        async for track in jamendo.iter_tracks(target=target, order="popularity_total"):
            try:
                resp = await http.get(
                    track.audio_url, headers={"Range": f"bytes=0-{PREVIEW_BYTES}"}
                )
                resp.raise_for_status()
                import librosa

                y, _ = librosa.load(io.BytesIO(resp.content), sr=SR, mono=True)
            except Exception as exc:
                logger.warning("skip %s: %s", track.id, exc)
                continue

            if y.size < WINDOW:
                continue

            offsets = window_offsets(y.size, WINDOW, 3)
            windows = [y[o : o + WINDOW] for o in offsets]

            clap_vecs.append(embed_track(windows))
            # Same windows for both, averaged the same way — the only
            # difference under test is the representation.
            lib_vecs.append(np.mean([librosa_features(w) for w in windows], axis=0))
            artists.append(track.artist_id or "?")
            tag_lists.append(track.tags)
            names.append(track.name)

            if len(clap_vecs) % 20 == 0:
                logger.info("collected %d", len(clap_vecs))

    return np.array(clap_vecs), np.array(lib_vecs), artists, tag_lists, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=int, default=120)
    args = parser.parse_args()

    clap, lib, artists, tags, _names = asyncio.run(collect(args.tracks))
    logger.info("collected %d tracks, %d distinct artists", len(clap), len(set(artists)))

    # CLAP output is already L2-normalized; librosa needs v1's treatment.
    lib_norm = _standardize_and_normalize(lib)

    rng = np.random.default_rng(0)
    results = {
        "CLAP (512d, pooled)": (
            pair_discrimination(clap, artists, np.random.default_rng(0)),
            tag_correlation(clap, tags),
        ),
        "librosa (56d, v1)": (
            pair_discrimination(lib_norm, artists, np.random.default_rng(0)),
            tag_correlation(lib_norm, tags),
        ),
    }
    del rng

    print()
    print(f"{'representation':<24}{'same-artist disc.':>20}{'tag correlation':>20}")
    print("-" * 64)
    for name, (disc, corr) in results.items():
        print(f"{name:<24}{disc:>19.1%}{corr:>20.3f}")
    print()
    print("same-artist discrimination: 50% = chance")
    print("tag correlation: Spearman rho vs. Jaccard tag overlap; 0 = no relationship")


if __name__ == "__main__":
    main()
