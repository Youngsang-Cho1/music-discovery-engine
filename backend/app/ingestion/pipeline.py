"""Ingestion pipeline: Jamendo → audio → CLAP + descriptors → Postgres + Qdrant.

Run as a module:

    python -m app.ingestion.pipeline --target 5000 --tags "electronic,ambient"

Audio is streamed, decoded, embedded, and discarded — only vectors and metadata
are persisted. v1 kept every downloaded .m4a on disk (git-ignored, tens of GB,
and not ours to store); here nothing copyrightable is retained, and the
playback path points at Jamendo's own CDN.
"""
import argparse
import asyncio
import logging

import httpx
import numpy as np
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app import cache
from app.config import get_settings
from app.db import Base, SessionLocal, engine
from app.embeddings.clap import embed_audio, load_waveform
from app.embeddings.descriptors import compute_descriptors
from app.ingestion.jamendo import JamendoClient, JamendoTrack
from app.models import Track
from app.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")

# CLAP only consumes a 10s window, so there is no reason to pull a whole 5MB
# track. Most CDNs honour Range; if this one doesn't we just get the full body.
PREVIEW_BYTES = 2_000_000
BATCH_SIZE = 16

# Shortest decode we will embed. Below a few seconds the CLAP vector describes
# a fragment rather than the track, and beat tracking has nothing to work with.
MIN_SECONDS = 3.0
MIN_SAMPLES = int(MIN_SECONDS * 48_000)


async def _download(client: httpx.AsyncClient, url: str) -> bytes | None:
    try:
        resp = await client.get(url, headers={"Range": f"bytes=0-{PREVIEW_BYTES}"})
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        logger.warning("download failed %s: %s", url, exc)
        return None


def _upsert_track(db, track: JamendoTrack, features: dict) -> None:
    """Insert-or-update one catalog row. Idempotent on re-ingest."""
    values = {
        "id": track.id,
        "name": track.name,
        "artist_id": track.artist_id,
        "artist_name": track.artist_name,
        "album_name": track.album_name,
        "duration_sec": track.duration_sec,
        "release_date": track.release_date,
        "audio_url": track.audio_url,
        "image_url": track.image_url,
        "license_url": track.license_url,
        "share_url": track.share_url,
        "tags": track.tags,
        "speed": track.speed,
        "acoustic_electric": track.acoustic_electric,
        "vocal_instrumental": track.vocal_instrumental,
        "lang": track.lang,
        "features": features,
        "embedded": True,
    }
    stmt = pg_insert(Track).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["id"],
        set_={k: v for k, v in values.items() if k != "id"},
    )
    db.execute(stmt)


async def ingest(target: int, tags: str | None, order: str) -> None:
    settings = get_settings()
    Base.metadata.create_all(bind=engine)

    store = VectorStore()
    store.ensure_collection()

    jamendo = JamendoClient()
    processed = skipped = 0

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http:
        batch: list[tuple[JamendoTrack, np.ndarray]] = []

        async for track in jamendo.iter_tracks(target=target, tags=tags, order=order):
            audio = await _download(http, track.audio_url)
            if not audio:
                skipped += 1
                continue
            try:
                waveform = load_waveform(audio)
            except Exception as exc:
                logger.warning("decode failed %s: %s", track.id, exc)
                skipped += 1
                continue

            # A too-short decode still embeds without raising, and the
            # resulting vector is indistinguishable from a real one once it is
            # in the index. Reject it here instead.
            if waveform.size < MIN_SAMPLES:
                logger.warning(
                    "decoded only %.1fs for %s, skipping",
                    waveform.size / settings.clap_sample_rate,
                    track.id,
                )
                skipped += 1
                continue

            batch.append((track, waveform))

            if len(batch) >= BATCH_SIZE:
                processed += _flush(batch, store, settings.clap_sample_rate)
                batch = []
                logger.info("ingested=%d skipped=%d", processed, skipped)

        if batch:
            processed += _flush(batch, store, settings.clap_sample_rate)

    # New tracks change what every ranking should return. Text-embedding cache
    # entries stay valid — the model didn't change, only the corpus.
    dropped = cache.invalidate("rank")
    logger.info(
        "done. ingested=%d skipped=%d vectors=%d rankings_invalidated=%d",
        processed,
        skipped,
        store.count(),
        dropped,
    )


def _flush(batch, store: VectorStore, sample_rate: int) -> int:
    """Embed one batch and write it to both stores."""
    tracks = [t for t, _ in batch]
    waveforms = [w for _, w in batch]

    vectors = embed_audio(waveforms)

    records = []
    with SessionLocal() as db:
        for track, waveform, vector in zip(tracks, waveforms, vectors, strict=True):
            features = compute_descriptors(waveform, sample_rate)
            _upsert_track(db, track, features)
            records.append(
                (
                    track.id,
                    vector,
                    {
                        "artist_id": track.artist_id,
                        "tags": track.tags,
                        "tempo_bpm": features.get("tempo_bpm"),
                        **{k: v for k, v in features.items() if k != "tempo_bpm"},
                    },
                )
            )
        db.commit()

    store.upsert(records)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CC-licensed tracks from Jamendo")
    parser.add_argument("--target", type=int, default=1000, help="tracks to ingest")
    parser.add_argument("--tags", type=str, default=None, help="comma-separated fuzzy tags")
    parser.add_argument(
        "--order",
        type=str,
        default="popularity_total",
        help="jamendo sort order (popularity_total, releasedate_desc, ...)",
    )
    args = parser.parse_args()
    asyncio.run(ingest(args.target, args.tags, args.order))


if __name__ == "__main__":
    main()
