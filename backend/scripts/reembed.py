"""Re-embed tracks already in the catalog with the current settings.

Changing how embeddings are produced — window count, pooling, or the model
itself — silently splits the index in two. Vectors built the old way and the
new way still sit in the same space and still return a cosine, so nothing
errors; the similarities are just wrong across the boundary. Measured for the
single-window → 3-window-pooled change, the same track's two vectors agree at
only ~0.56-0.89 cosine, which is the same range as genuinely different tracks.

Re-ingestion through the normal pipeline doesn't fix it, because that only
touches tracks the Jamendo query happens to return. This walks Postgres
instead, so every stored track gets refreshed.

    python -m scripts.reembed --limit 1000
    python -m scripts.reembed --dry-run
"""
import argparse
import asyncio
import logging

import httpx
from sqlalchemy import select

from app import cache
from app.config import get_settings
from app.db import SessionLocal
from app.embeddings.clap import embed_track, load_windows
from app.embeddings.descriptors import compute_descriptors
from app.ingestion.pipeline import MIN_SAMPLES, PREVIEW_BYTES
from app.models import Track
from app.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reembed")
logging.getLogger("httpx").setLevel(logging.WARNING)

BATCH_SIZE = 16


async def _fetch(http: httpx.AsyncClient, track_id: str, url: str):
    try:
        resp = await http.get(url, headers={"Range": f"bytes=0-{PREVIEW_BYTES}"})
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("download failed %s: %s", track_id, exc)
        return None

    try:
        windows = await asyncio.to_thread(load_windows, resp.content)
    except Exception as exc:
        logger.warning("decode failed %s: %s", track_id, exc)
        return None

    if not windows or windows[len(windows) // 2].size < MIN_SAMPLES:
        return None
    return track_id, windows


async def reembed(limit: int | None, dry_run: bool) -> None:
    settings = get_settings()
    store = VectorStore()
    store.ensure_collection()

    with SessionLocal() as db:
        stmt = select(Track.id, Track.audio_url, Track.artist_id, Track.tags).where(
            Track.audio_url.isnot(None)
        )
        if limit:
            stmt = stmt.limit(limit)
        rows = db.execute(stmt).all()

    logger.info(
        "re-embedding %d tracks with %d windows each",
        len(rows),
        settings.clap_windows_per_track,
    )
    if dry_run:
        return

    updated = failed = 0

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http:
        for start in range(0, len(rows), BATCH_SIZE):
            chunk = rows[start : start + BATCH_SIZE]
            fetched = await asyncio.gather(
                *(_fetch(http, r.id, r.audio_url) for r in chunk)
            )

            by_id = {r.id: r for r in chunk}
            records = []
            with SessionLocal() as db:
                for item in fetched:
                    if item is None:
                        failed += 1
                        continue
                    track_id, windows = item
                    row = by_id[track_id]

                    vector = embed_track(windows)
                    features = compute_descriptors(
                        windows[len(windows) // 2], settings.clap_sample_rate
                    )

                    # Descriptors are recomputed too: they come from the same
                    # window selection, so leaving them stale would make the
                    # `descriptor` ranking signal disagree with the vector.
                    db.query(Track).filter(Track.id == track_id).update(
                        {"features": features}
                    )
                    records.append(
                        (
                            track_id,
                            vector,
                            {
                                "artist_id": row.artist_id,
                                "tags": row.tags or [],
                                **features,
                            },
                        )
                    )
                db.commit()

            if records:
                store.upsert(records)
                updated += len(records)

            logger.info("updated=%d failed=%d / %d", updated, failed, len(rows))

    cache.invalidate("rank")
    logger.info("done. updated=%d failed=%d", updated, failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-embed stored tracks")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(reembed(args.limit, args.dry_run))


if __name__ == "__main__":
    main()
