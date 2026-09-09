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
from app.embeddings.clap import embed_track, load_windows
from app.embeddings.descriptors import compute_descriptors
from app.ingestion.jamendo import JamendoClient, JamendoTrack
from app.models import Track
from app.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")

# Enough bytes for the sampled windows to span a real portion of the track.
# 2MB decoded to ~70-125s, which put every window inside the first two minutes;
# windows that never reach the second half describe an intro, not a song.
# Most CDNs honour Range; if this one doesn't we just get the full body.
PREVIEW_BYTES = 6_000_000
# Also the download concurrency: a batch is fetched in parallel, then embedded
# in one CLAP call. Larger batches hide more network latency but hold more
# decoded audio in memory at once.
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
        "listens": track.listens,
        "favorites": track.favorites,
        "features": features,
        "embedded": True,
    }
    stmt = pg_insert(Track).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["id"],
        set_={k: v for k, v in values.items() if k != "id"},
    )
    db.execute(stmt)


async def _prepare(
    http: httpx.AsyncClient, track: JamendoTrack, sample_rate: int
) -> tuple[JamendoTrack, np.ndarray, list[np.ndarray]] | None:
    """Download and decode one track. None means skip it."""
    audio = await _download(http, track.audio_url)
    if not audio:
        return None

    try:
        # librosa decoding is CPU-bound and releases the GIL poorly; run it off
        # the event loop so it doesn't stall the concurrent downloads.
        windows = await asyncio.to_thread(load_windows, audio)
    except Exception as exc:
        logger.warning("decode failed %s: %s", track.id, exc)
        return None

    if not windows:
        logger.warning("decoded nothing for %s, skipping", track.id)
        return None

    # Descriptors read the middle window: they describe absolute properties
    # (tempo, loudness) where averaging across sections would smear a tempo
    # change into a value the track never has.
    waveform = windows[len(windows) // 2]

    # A too-short decode still embeds without raising, and the resulting vector
    # is indistinguishable from a real one once it is in the index.
    if waveform.size < MIN_SAMPLES:
        logger.warning(
            "decoded only %.1fs for %s, skipping", waveform.size / sample_rate, track.id
        )
        return None

    return track, waveform, windows


async def ingest_once(
    target: int, tags: str | None, order: str, store: VectorStore, jamendo: JamendoClient
) -> tuple[int, int]:
    settings = get_settings()
    processed = skipped = 0

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http:
        pending: list[JamendoTrack] = []

        async def drain() -> tuple[int, int]:
            """Fetch a batch concurrently, then embed it in one CLAP call."""
            # Downloads dominated wall time when done one at a time — each track
            # is a few MB over the network while the CPU sits idle.
            results = await asyncio.gather(
                *(_prepare(http, t, settings.clap_sample_rate) for t in pending)
            )
            ready = [r for r in results if r is not None]
            failed = len(results) - len(ready)
            if not ready:
                return 0, failed
            return _flush(ready, store, settings.clap_sample_rate), failed

        async for track in jamendo.iter_tracks(target=target, tags=tags, order=order):
            pending.append(track)
            if len(pending) >= BATCH_SIZE:
                done, failed = await drain()
                processed += done
                skipped += failed
                pending = []
                logger.info("ingested=%d skipped=%d", processed, skipped)

        if pending:
            done, failed = await drain()
            processed += done
            skipped += failed

    return processed, skipped


async def sweep(
    per_combo: int, tag_list: list[str], order_list: list[str]
) -> None:
    """Ingest every (tag, order) combination in one process.

    Two reasons this is not a shell loop over the CLI:

    * CLAP takes ~26s to load. Re-running the module per combination paid that
      once per combination — 13 minutes of pure model loading across a 30-combo
      sweep, which dominated the actual work.
    * Sorting by popularity always returns the same head of the catalog no
      matter which tag is attached. Measured: 0 new rows out of 100 once that
      head is ingested, against ~100% new for release-date or id ordering.
      Diversity comes from varying the sort, so the sweep is over both axes.
    """
    Base.metadata.create_all(bind=engine)

    store = VectorStore()
    store.ensure_collection()
    jamendo = JamendoClient()

    total_new = total_skipped = 0
    before = store.count()

    combos_failed = 0

    for order in order_list:
        for tag in tag_list:
            # A sweep is a long sequence of independent units of work. Letting
            # one propagate loses every combination after it — a single
            # transient DNS failure once cost 13 of 14 genres. Each combination
            # is isolated so the run degrades instead of aborting.
            try:
                done, failed = await ingest_once(per_combo, tag, order, store, jamendo)
            except Exception:
                combos_failed += 1
                logger.exception("[%s/%s] failed, continuing with the next combination", tag, order)
                continue

            total_new += done
            total_skipped += failed
            logger.info(
                "[%s/%s] ingested=%d skipped=%d | running total=%d vectors=%d",
                tag,
                order,
                done,
                failed,
                total_new,
                store.count(),
            )

    # New tracks change what every ranking should return. Text-embedding cache
    # entries stay valid — the model didn't change, only the corpus.
    dropped = cache.invalidate("rank")
    logger.info(
        "sweep done. processed=%d skipped=%d combos_failed=%d vectors %d -> %d "
        "rankings_invalidated=%d",
        total_new,
        total_skipped,
        combos_failed,
        before,
        store.count(),
        dropped,
    )


def _flush(batch, store: VectorStore, sample_rate: int) -> int:
    """Embed one batch and write it to both stores."""
    tracks = [t for t, _, _ in batch]
    waveforms = [w for _, w, _ in batch]

    # Pooled per track: each track's windows are embedded together and averaged.
    vectors = [embed_track(windows) for _, _, windows in batch]

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
                        "listens": track.listens,
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
    parser.add_argument(
        "--target", type=int, default=500, help="tracks to fetch per (tag, order) combination"
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="",
        help="comma-separated tags, each swept separately (empty = no tag filter)",
    )
    parser.add_argument(
        "--orders",
        type=str,
        default="releasedate_desc",
        help=(
            "comma-separated jamendo sort orders. popularity_total returns the same "
            "global head for every tag; releasedate_desc / id_asc / name reach the rest."
        ),
    )
    args = parser.parse_args()

    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()] or [None]
    order_list = [o.strip() for o in args.orders.split(",") if o.strip()]
    asyncio.run(sweep(args.target, tag_list, order_list))


if __name__ == "__main__":
    main()
