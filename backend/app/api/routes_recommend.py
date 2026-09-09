"""Recommendation endpoints.

Every response carries the `variant` that produced it. The client echoes that
value back on each interaction event, which is what lets the bandit attribute
outcomes to arms without server-side session state.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_store
from app.api.rate_limit import READ_BUDGET, SEARCH_BUDGET, rate_limit
from app.db import get_db
from app.embeddings.text_encoder import TextSearchUnavailable
from app.experiments import bandit
from app.models import Track
from app.retrieval.vector_store import VectorStore
from app.schemas import RecommendationOut, RecommendationResponse, TrackOut
from app.services import recommender

router = APIRouter(prefix="/recommend", tags=["recommend"])


def _parse_tempo(tempo_min: float | None, tempo_max: float | None):
    if tempo_min is None and tempo_max is None:
        return None
    return (tempo_min or 0.0, tempo_max or 400.0)


def _to_out(recs) -> list[RecommendationOut]:
    return [
        RecommendationOut(
            track=TrackOut.model_validate(r.track),
            relevance=r.relevance,
            signals=r.signals,
        )
        for r in recs
    ]


@router.get(
    "/track/{track_id}",
    response_model=RecommendationResponse,
    dependencies=[Depends(rate_limit(READ_BUDGET))],
)
def recommend_from_track(
    track_id: str,
    db: Session = Depends(get_db),
    store: VectorStore = Depends(get_store),
    session_id: str | None = None,
    force_variant: str | None = Query(None, description="Bypass bandit assignment (QA)"),
    top_k: int = Query(20, le=50),
    tags: list[str] | None = Query(None),
    tempo_min: float | None = None,
    tempo_max: float | None = None,
):
    session_id = session_id or str(uuid.uuid4())
    variant = bandit.choose_variant(db, forced=force_variant)

    recs = recommender.recommend_by_track(
        db,
        store,
        seed_track_id=track_id,
        variant=variant,
        top_k=top_k,
        tags=tags,
        tempo_range=_parse_tempo(tempo_min, tempo_max),
    )
    if not recs and db.get(Track, track_id) is None:
        raise HTTPException(status_code=404, detail="track not found")

    bandit.record_impression(db, variant, count=len(recs))
    return RecommendationResponse(
        variant=variant,
        session_id=session_id,
        seed_track_id=track_id,
        results=_to_out(recs),
    )


# Tighter budget: an uncached query costs a transformer forward pass.
@router.get(
    "/text",
    response_model=RecommendationResponse,
    dependencies=[Depends(rate_limit(SEARCH_BUDGET))],
)
def recommend_from_text(
    q: str = Query(..., min_length=2, description="e.g. 'rainy day lo-fi with soft piano'"),
    db: Session = Depends(get_db),
    store: VectorStore = Depends(get_store),
    session_id: str | None = None,
    force_variant: str | None = None,
    top_k: int = Query(20, le=50),
    tags: list[str] | None = Query(None),
    tempo_min: float | None = None,
    tempo_max: float | None = None,
):
    """Natural-language mood search against the CLAP index."""
    session_id = session_id or str(uuid.uuid4())
    variant = bandit.choose_variant(db, forced=force_variant)

    try:
        recs = recommender.recommend_by_text(
            db,
            store,
            query=q,
            variant=variant,
            top_k=top_k,
            tags=tags,
            tempo_range=_parse_tempo(tempo_min, tempo_max),
        )
    except TextSearchUnavailable as exc:
        # 503, not 500: the rest of the API is healthy and seed-track
        # recommendations still work.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    bandit.record_impression(db, variant, count=len(recs))
    return RecommendationResponse(
        variant=variant,
        session_id=session_id,
        query=q,
        results=_to_out(recs),
    )
