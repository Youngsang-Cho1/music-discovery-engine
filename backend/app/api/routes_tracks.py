"""Catalog browsing endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Track
from app.schemas import TrackOut

router = APIRouter(prefix="/tracks", tags=["tracks"])


@router.get("", response_model=list[TrackOut])
def list_tracks(
    db: Session = Depends(get_db),
    limit: int = Query(50, le=200),
    offset: int = 0,
    q: str | None = None,
):
    """List or keyword-search the catalog.

    This is plain metadata lookup for the browse UI — semantic search lives on
    /recommend/text and goes through CLAP instead.
    """
    stmt = select(Track).where(Track.embedded.is_(True))
    if q:
        pattern = f"%{q}%"
        stmt = stmt.where(or_(Track.name.ilike(pattern), Track.artist_name.ilike(pattern)))
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
