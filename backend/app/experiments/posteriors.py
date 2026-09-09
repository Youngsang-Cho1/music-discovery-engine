"""Beta-posterior math for the bandit, kept free of the database.

Separated from `bandit.py` so the statistics can be unit-tested against known
distributions instead of only through a live Postgres session.
"""
from dataclasses import dataclass

import numpy as np

# Guards against a degenerate Beta(0, ·) if an arm's counters are ever zeroed.
_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class Posterior:
    variant: str
    alpha: float
    beta: float

    @property
    def safe(self) -> tuple[float, float]:
        return max(self.alpha, _EPS), max(self.beta, _EPS)

    @property
    def mean(self) -> float:
        a, b = self.safe
        return a / (a + b)


def thompson_pick(posteriors: list[Posterior], rng: np.random.Generator | None = None) -> str:
    """Draw once from each posterior and return the variant with the top draw.

    This is the whole of Thompson sampling: an arm is chosen with exactly the
    probability that it is optimal under current belief, so exploration decays
    on its own as posteriors tighten — no epsilon to schedule by hand.
    """
    if not posteriors:
        raise ValueError("no posteriors to sample from")

    rng = rng or np.random.default_rng()
    draws = [rng.beta(*p.safe) for p in posteriors]
    return posteriors[int(np.argmax(draws))].variant


def credible_interval(posterior: Posterior, level: float = 0.95) -> tuple[float, float]:
    """Equal-tailed Beta credible interval."""
    from scipy import stats

    a, b = posterior.safe
    tail = (1.0 - level) / 2
    lo, hi = stats.beta.ppf([tail, 1 - tail], a, b)
    return float(lo), float(hi)


def probability_best(
    posteriors: list[Posterior],
    draws: int = 20_000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Monte-Carlo P(arm has the highest true rate).

    Comparing posterior means alone would call a 2-point gap on 30 impressions
    a winner; this integrates over the uncertainty in every arm at once, which
    is the quantity a ship/no-ship decision should actually rest on.
    """
    if not posteriors:
        return {}

    rng = rng or np.random.default_rng(0)
    samples = np.column_stack([rng.beta(*p.safe, draws) for p in posteriors])
    winners = samples.argmax(axis=1)
    return {p.variant: float((winners == i).mean()) for i, p in enumerate(posteriors)}
