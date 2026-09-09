"""Thompson-sampling bandit over ranking variants.

A fixed A/B split wastes traffic on arms that are already visibly losing, and
v1's alternative — one lambda chosen offline against a same-artist proxy — never
observed a real listener at all. Thompson sampling keeps every arm alive with
probability proportional to how likely it is to be best, so exploration cost
shrinks automatically as evidence accumulates.

Reward is binary per impression: 1 if the listener kept the track past the skip
threshold, 0 otherwise. Beta(alpha, beta) is the conjugate posterior for that.
"""
import logging
import random

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.experiments.variants import DEFAULT_VARIANT, VARIANTS
from app.models import BanditArm

logger = logging.getLogger(__name__)

# A play shorter than this counts as a rejection of the recommendation.
SKIP_THRESHOLD_MS = 10_000


def ensure_arms(db: Session) -> None:
    """Insert a Beta(1, 1) prior for any variant not yet in the table."""
    for name in VARIANTS:
        stmt = (
            pg_insert(BanditArm)
            .values(variant=name, alpha=1.0, beta=1.0, impressions=0, rewards=0)
            .on_conflict_do_nothing(index_elements=["variant"])
        )
        db.execute(stmt)
    db.commit()


def load_arms(db: Session) -> list[BanditArm]:
    return list(db.execute(select(BanditArm)).scalars())


def choose_variant(db: Session, *, forced: str | None = None) -> str:
    """Sample one arm from its posterior and return the winner's name.

    `forced` bypasses sampling — used for held-out QA sessions and for
    reproducing a specific ranking in a bug report.
    """
    if forced and forced in VARIANTS:
        return forced

    arms = load_arms(db)
    if not arms:
        return DEFAULT_VARIANT

    # One draw per arm; the largest draw wins. Arms with little data have wide
    # posteriors and therefore still win often enough to keep being explored.
    best_name, best_draw = DEFAULT_VARIANT, -1.0
    for arm in arms:
        if arm.variant not in VARIANTS:
            continue
        draw = random.betavariate(max(arm.alpha, 1e-6), max(arm.beta, 1e-6))
        if draw > best_draw:
            best_name, best_draw = arm.variant, draw
    return best_name


def record_impression(db: Session, variant: str, count: int = 1) -> None:
    arm = db.get(BanditArm, variant)
    if arm is None:
        return
    arm.impressions += count
    db.commit()


def record_outcome(db: Session, variant: str, *, reward: bool) -> None:
    """Update the Beta posterior with one observed outcome."""
    arm = db.get(BanditArm, variant)
    if arm is None:
        return
    if reward:
        arm.alpha += 1.0
        arm.rewards += 1
    else:
        arm.beta += 1.0
    db.commit()


def arm_summary(db: Session) -> list[dict]:
    """Posterior mean and 95% credible interval per arm, for the dashboard."""
    from scipy import stats  # optional dependency; imported lazily

    out = []
    for arm in load_arms(db):
        a, b = max(arm.alpha, 1e-6), max(arm.beta, 1e-6)
        lo, hi = stats.beta.ppf([0.025, 0.975], a, b)
        out.append(
            {
                "variant": arm.variant,
                "impressions": arm.impressions,
                "rewards": arm.rewards,
                "posterior_mean": float(a / (a + b)),
                "ci_low": float(lo),
                "ci_high": float(hi),
            }
        )
    out.sort(key=lambda r: r["posterior_mean"], reverse=True)
    return out


def probability_best(db: Session, draws: int = 20_000) -> dict[str, float]:
    """Monte-Carlo P(arm is best) — the number to actually ship a decision on.

    A raw conversion-rate comparison ignores how much of the gap is noise;
    this integrates over both posteriors instead.
    """
    arms = load_arms(db)
    if not arms:
        return {}

    rng = np.random.default_rng(0)
    samples = np.column_stack(
        [rng.beta(max(a.alpha, 1e-6), max(a.beta, 1e-6), draws) for a in arms]
    )
    winners = samples.argmax(axis=1)
    return {
        arm.variant: float((winners == i).mean()) for i, arm in enumerate(arms)
    }
