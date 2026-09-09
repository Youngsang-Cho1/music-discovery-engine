import numpy as np
import pytest

from app.experiments.posteriors import (
    Posterior,
    credible_interval,
    probability_best,
    thompson_pick,
)


def test_mean_matches_beta_expectation():
    assert Posterior("a", 3.0, 1.0).mean == pytest.approx(0.75)


def test_zero_counts_do_not_blow_up():
    """A zeroed arm must degrade to a valid distribution, not a ZeroDivisionError."""
    assert 0.0 < Posterior("a", 0.0, 0.0).mean < 1.0
    lo, hi = credible_interval(Posterior("a", 0.0, 0.0))
    assert 0.0 <= lo <= hi <= 1.0


def test_credible_interval_tightens_with_evidence():
    wide = credible_interval(Posterior("a", 2, 2))
    tight = credible_interval(Posterior("a", 200, 200))
    assert (tight[1] - tight[0]) < (wide[1] - wide[0])


def test_thompson_pick_favours_the_better_arm():
    """With a large evidence gap, sampling should land on the better arm most
    of the time — but not every time, or it would have stopped exploring."""
    rng = np.random.default_rng(42)
    arms = [Posterior("weak", 5, 95), Posterior("strong", 60, 40)]

    picks = [thompson_pick(arms, rng) for _ in range(2000)]
    strong_share = picks.count("strong") / len(picks)
    assert strong_share > 0.95


def test_thompson_pick_still_explores_under_uncertainty():
    """Two untouched arms should split traffic roughly evenly."""
    rng = np.random.default_rng(7)
    arms = [Posterior("a", 1, 1), Posterior("b", 1, 1)]

    picks = [thompson_pick(arms, rng) for _ in range(4000)]
    share = picks.count("a") / len(picks)
    assert 0.45 < share < 0.55


def test_thompson_pick_rejects_empty():
    with pytest.raises(ValueError):
        thompson_pick([])


def test_probability_best_sums_to_one():
    arms = [Posterior("a", 30, 70), Posterior("b", 40, 60), Posterior("c", 10, 90)]
    p = probability_best(arms, draws=5000)
    assert set(p) == {"a", "b", "c"}
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-9)


def test_probability_best_is_not_fooled_by_a_small_sample():
    """5/10 beats 400/1000 on raw rate, but nowhere near confidently.

    This is exactly the case a naive conversion-rate comparison gets wrong,
    and the reason the dashboard reports P(best) instead of the mean.
    """
    small_lead = Posterior("small", 6, 6)      # 5/10 observed, mean 0.5
    big_sample = Posterior("large", 401, 601)  # 400/1000 observed, mean 0.4
    p = probability_best([small_lead, big_sample], draws=20000)
    assert p["small"] < 0.95


def test_probability_best_empty():
    assert probability_best([]) == {}
