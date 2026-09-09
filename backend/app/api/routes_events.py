"""Interaction logging + experiment reporting.

Events are the product's only ground truth. The offline pair-discrimination
metric v1 optimised (does a same-artist pair rank above a random pair?) is a
proxy nobody listens to; skip-vs-complete is the thing itself.
"""
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.experiments import bandit
from app.models import InteractionEvent
from app.schemas import ArmStat, EventIn, ExperimentReport

router = APIRouter(tags=["events"])

# Terminal actions that resolve an impression into a reward or a non-reward.
_REWARD_ACTIONS = {"complete", "like"}
_PENALTY_ACTIONS = {"skip"}


@router.post("/events", status_code=202)
def log_event(event: EventIn, db: Session = Depends(get_db)):
    """Record one interaction and, if it resolves an impression, update the arm."""
    db.add(
        InteractionEvent(
            session_id=event.session_id,
            variant=event.variant,
            seed_track_id=event.seed_track_id,
            track_id=event.track_id,
            action=event.action,
            position=event.position,
            dwell_ms=event.dwell_ms,
        )
    )
    db.commit()

    if event.action in _REWARD_ACTIONS:
        bandit.record_outcome(db, event.variant, reward=True)
    elif event.action in _PENALTY_ACTIONS:
        # A skip only counts against the arm if it was fast. A listener who
        # skips at 2:30 heard most of the track and did not reject the pick.
        dwell = event.dwell_ms or 0
        if dwell < bandit.SKIP_THRESHOLD_MS:
            bandit.record_outcome(db, event.variant, reward=False)

    return {"status": "accepted"}


@router.get("/experiments/report", response_model=ExperimentReport)
def experiment_report(db: Session = Depends(get_db)):
    """Per-arm posteriors plus P(best) — the public experiment dashboard."""
    summary = bandit.arm_summary(db)
    p_best = bandit.probability_best(db)

    arms = [ArmStat(**row, probability_best=p_best.get(row["variant"])) for row in summary]
    return ExperimentReport(
        arms=arms,
        total_impressions=sum(a.impressions for a in arms),
        total_rewards=sum(a.rewards for a in arms),
    )


@router.get("/experiments/events/summary")
def event_summary(db: Session = Depends(get_db)):
    """Raw event counts by variant and action, for sanity-checking the arms."""
    rows = db.execute(
        select(
            InteractionEvent.variant,
            InteractionEvent.action,
            func.count().label("n"),
            func.avg(InteractionEvent.dwell_ms).label("avg_dwell_ms"),
        ).group_by(InteractionEvent.variant, InteractionEvent.action)
    ).all()

    return [
        {
            "variant": r.variant,
            "action": r.action,
            "count": r.n,
            "avg_dwell_ms": float(r.avg_dwell_ms) if r.avg_dwell_ms is not None else None,
        }
        for r in rows
    ]
