"""Seed a synthetic catalog so the stack can be run without a Jamendo key.

Vectors are drawn around a handful of cluster centroids rather than uniformly,
because uniform random vectors in 512 dimensions are all roughly equidistant —
a recommender over them returns arbitrary results and proves nothing. Clusters
give the ANN index real neighbourhood structure to find.

    python -m scripts.seed_demo --tracks 300

Not a substitute for real ingestion: no audio, and the "embeddings" carry no
acoustic meaning. It exists to exercise wiring, not quality.
"""
import argparse

import numpy as np

from app import cache
from app.config import get_settings
from app.db import Base, SessionLocal, engine
from app.embeddings.descriptors import compute_descriptors
from app.models import Track
from app.retrieval.vector_store import VectorStore

GENRES = [
    ("ambient", ["ambient", "atmospheric", "synth"], 70.0),
    ("techno", ["techno", "electronic", "kick"], 132.0),
    ("acoustic", ["acoustic", "guitar", "folk"], 95.0),
    ("jazz", ["jazz", "piano", "brass"], 110.0),
    ("cinematic", ["cinematic", "strings", "orchestral"], 80.0),
]

SR = 22_050


def _synthetic_waveform(rng: np.random.Generator, tempo: float) -> np.ndarray:
    """A tone plus a click track, so descriptors come out of the real code path."""
    seconds = 8.0
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    freq = float(rng.uniform(120, 2200))
    signal = 0.4 * np.sin(2 * np.pi * freq * t)

    step = int(SR * 60.0 / tempo)
    for i in range(0, len(signal) - 64, step):
        signal[i : i + 64] += np.hanning(64)

    return signal.astype(np.float32)


def seed(n_tracks: int, seed_value: int = 0) -> None:
    settings = get_settings()
    rng = np.random.default_rng(seed_value)

    Base.metadata.create_all(bind=engine)
    store = VectorStore()
    store.ensure_collection()

    dim = settings.embedding_dim
    centroids = rng.normal(size=(len(GENRES), dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    records = []
    with SessionLocal() as db:
        for i in range(n_tracks):
            g = i % len(GENRES)
            genre, tags, base_tempo = GENRES[g]
            tempo = float(np.clip(rng.normal(base_tempo, 8.0), 50, 190))

            # Cluster centroid + noise, renormalized back onto the unit sphere
            # so cosine distance behaves the same way it does for CLAP output.
            vec = centroids[g] + rng.normal(scale=0.35, size=dim).astype(np.float32)
            vec /= np.linalg.norm(vec)

            features = compute_descriptors(_synthetic_waveform(rng, tempo), SR)
            # Several tracks per artist, so the per-artist cap has something to cap.
            artist_id = f"artist-{genre}-{i % 7}"

            track = Track(
                id=f"demo-{i:05d}",
                name=f"{genre.title()} Study No. {i // len(GENRES) + 1}",
                artist_id=artist_id,
                artist_name=artist_id.replace("-", " ").title(),
                album_name=f"{genre.title()} Sessions",
                duration_sec=int(rng.integers(120, 300)),
                audio_url=None,
                image_url=None,
                license_url="https://creativecommons.org/licenses/by/3.0/",
                tags=tags,
                features=features,
                embedded=True,
            )
            db.merge(track)

            records.append(
                (
                    track.id,
                    vec,
                    {
                        "artist_id": artist_id,
                        "tags": tags,
                        # Skewed play counts give the novelty signal a real
                        # long tail to prefer.
                        "listens": int(rng.pareto(1.2) * 100),
                        **features,
                    },
                )
            )
        db.commit()

    store.upsert(records)
    cache.invalidate("rank")
    print(f"seeded {len(records)} tracks; qdrant now holds {store.count()} vectors")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a synthetic demo catalog")
    parser.add_argument("--tracks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed(args.tracks, args.seed)


if __name__ == "__main__":
    main()
