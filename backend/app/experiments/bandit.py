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

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.experiments import posteriors as post
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


def _live_posteriors(db: Session) -> list[post.Posterior]:
    """Posteriors for arms that still exist in the variant registry.

    A retired variant keeps its row for historical event joins, but must not
    keep receiving traffic.
    """
    return [
        post.Posterior(arm.variant, arm.alpha, arm.beta)
        for arm in load_arms(db)
        if arm.variant in VARIANTS
    ]


def choose_variant(db: Session, *, forced: str | None = None) -> str:
    """Sample one arm from its posterior and return the winner's name.

    `forced` bypasses sampling — used for held-out QA sessions and for
    reproducing a specific ranking in a bug report.
    """
    if forced and forced in VARIANTS:
        return forced

    live = _live_posteriors(db)
    if not live:
        return DEFAULT_VARIANT
    return post.thompson_pick(live)


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
    out = []
    for arm in load_arms(db):
        posterior = post.Posterior(arm.variant, arm.alpha, arm.beta)
        lo, hi = post.credible_interval(posterior)
        out.append(
            {
                "variant": arm.variant,
                "impressions": arm.impressions,
                "rewards": arm.rewards,
                "posterior_mean": posterior.mean,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    out.sort(key=lambda r: r["posterior_mean"], reverse=True)
    return out


def probability_best(db: Session, draws: int = 20_000) -> dict[str, float]:
    """Monte-Carlo P(arm is best) — the number to actually ship a decision on."""
    return post.probability_best(_live_posteriors(db), draws=draws)
