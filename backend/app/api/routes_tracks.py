"""Catalog browsing endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import cast, func, or_, select
from sqlalchemy.dialects.postgresql import JSONB, array
from sqlalchemy.orm import Session

from app.api.rate_limit import READ_BUDGET, rate_limit
from app.db import get_db
from app.models import Track
from app.schemas import TrackOut

router = APIRouter(
    prefix="/tracks",
    tags=["tracks"],
    dependencies=[Depends(rate_limit(READ_BUDGET))],
)


@router.get("", response_model=list[TrackOut])
def list_tracks(
    db: Session = Depends(get_db),
    limit: int = Query(50, le=200),
    offset: int = 0,
    q: str | None = Query(None, description="Substring match on track or artist name"),
    tags: list[str] | None = Query(None, description="Keep tracks carrying any of these tags"),
):
    """List, keyword-search, or tag-filter the catalog.

    Plain metadata lookup for the browse UI — semantic search lives on
    /recommend/text and goes through CLAP instead.

    Tag filtering happens here rather than in the client because the client
    only ever holds one page: filtering there silently searches the first 100
    rows and reports "no tracks" for a tag that has hundreds.
    """
    stmt = select(Track).where(Track.embedded.is_(True))

    if q:
        pattern = f"%{q}%"
        stmt = stmt.where(or_(Track.name.ilike(pattern), Track.artist_name.ilike(pattern)))

    if tags:
        # `tags` is a JSON array column; ?| asks whether it contains any of the
        # given keys. Cast is needed because the column is json, not jsonb.
        stmt = stmt.where(cast(Track.tags, JSONB).op("?|")(array(tags)))

    stmt = stmt.order_by(Track.name).limit(limit).offset(offset)
    return list(db.execute(stmt).scalars())


@router.get("/random", response_model=list[TrackOut])
def random_tracks(db: Session = Depends(get_db), limit: int = Query(12, le=50)):
    """Random embedded tracks — used to seed the landing page."""
    stmt = (
        select(Track)
        .where(Track.embedded.is_(True))
        .order_by(func.random())
        .limit(limit)
    )
    return list(db.execute(stmt).scalars())


@router.get("/{track_id}", response_model=TrackOut)
def get_track(track_id: str, db: Session = Depends(get_db)):
    track = db.get(Track, track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="track not found")
    return track
