"""Catalog-level statistics.

Separate from `/tracks` because these are aggregates over the whole corpus, not
a page of rows. The UI uses them to say what is actually in the index — a
discovery engine that can't tell you the shape of its own catalog is asking for
trust it hasn't earned.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.api.rate_limit import READ_BUDGET, rate_limit
from app.db import get_db
from app.models import Track
from app.schemas import CatalogStats, TagCount

router = APIRouter(
    prefix="/catalog",
    tags=["catalog"],
    dependencies=[Depends(rate_limit(READ_BUDGET))],
)


@router.get("/stats", response_model=CatalogStats)
def catalog_stats(db: Session = Depends(get_db), top_tags: int = Query(24, le=100)):
    """Corpus size, artist count, and the most common tags."""
    totals = db.execute(
        select(
            func.count(Track.id),
            func.count(func.distinct(Track.artist_id)),
            func.coalesce(func.sum(Track.duration_sec), 0),
        ).where(Track.embedded.is_(True))
    ).one()

    # jsonb_array_elements_text unnests the tag arrays so they can be grouped.
    # Doing this in SQL rather than pulling every row into Python keeps it a
    # single round trip as the catalog grows.
    rows = db.execute(
        text(
            """
            SELECT tag, count(*) AS n
            FROM tracks, jsonb_array_elements_text(tags::jsonb) AS tag
            WHERE embedded IS TRUE
            GROUP BY tag
            ORDER BY n DESC
            LIMIT :limit
            """
        ),
        {"limit": top_tags},
    ).all()

    return CatalogStats(
        tracks=totals[0],
        artists=totals[1],
        total_duration_sec=int(totals[2] or 0),
        top_tags=[TagCount(tag=r.tag, count=r.n) for r in rows],
    )
