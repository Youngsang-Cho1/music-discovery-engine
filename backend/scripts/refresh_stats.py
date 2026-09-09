"""Backfill play counts for tracks already in the catalog.

The `novelty` ranking signal is `1 / (1 + log1p(listens))`. Until play counts
were being captured, every track had `listens = 0`, so novelty was exactly 1.0
for all of them — the `discovery` arm was ranking identically to a plain
similarity arm, and the long-tail claim was not actually being tested.

Fixing the ingestion path only helps tracks ingested after the fix. This
refreshes the rest, fetching metadata by id in bulk: no audio download, no
re-embedding, ~200 tracks per request.

    python -m scripts.refresh_stats
"""
import argparse
import asyncio
import logging

from sqlalchemy import select, update

from app import cache
from app.db import SessionLocal
from app.ingestion.jamendo import MAX_LIMIT, JamendoClient
from app.models import Track
from app.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("refresh-stats")
logging.getLogger("httpx").setLevel(logging.WARNING)


async def refresh(dry_run: bool) -> None:
    store = VectorStore()
    jamendo = JamendoClient()

    with SessionLocal() as db:
        rows = db.execute(select(Track.id, Track.artist_id, Track.tags, Track.features)).all()

    logger.info("refreshing stats for %d tracks", len(rows))
    if dry_run:
        return

    by_id = {r.id: r for r in rows}
    ids = list(by_id)
    updated = missing = 0

    for start in range(0, len(ids), MAX_LIMIT):
        chunk = ids[start : start + MAX_LIMIT]

        # Fetch by id — cheap metadata lookup, no audio.
        fetched = await jamendo.fetch_stats(chunk)

        records = []
        with SessionLocal() as db:
            for track_id in chunk:
                stats = fetched.get(track_id)
                if stats is None:
                    missing += 1
                    continue

                listens = int(stats.get("rate_listened_total") or 0)
                favorites = int(stats.get("favorited") or 0)
                db.execute(
                    update(Track)
                    .where(Track.id == track_id)
                    .values(listens=listens, favorites=favorites)
                )

                # The payload is what the reranker actually reads, so updating
                # Postgres alone would leave novelty stuck at 1.0.
                row = by_id[track_id]
                records.append(
                    (
                        track_id,
                        None,
                        {
                            "artist_id": row.artist_id,
                            "tags": row.tags or [],
                            "listens": listens,
                            **(row.features or {}),
                        },
                    )
                )
                updated += 1
            db.commit()

        if records:
            store.set_payloads([(tid, payload) for tid, _, payload in records])

        logger.info("updated=%d missing=%d / %d", updated, missing, len(ids))

    cache.invalidate("rank")
    logger.info("done. updated=%d missing=%d", updated, missing)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Jamendo play counts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(refresh(args.dry_run))


if __name__ == "__main__":
    main()
