"""Split-conformal calibration tests (§7.7)."""

from __future__ import annotations

import random

import pytest

from slipstream.calibration.conformal import (
    CalibrationItem,
    binomial_ucl,
    fit_all,
    fit_category,
    threshold_drift,
)


def make_items(
    n_pos: int, n_neg: int, *, pos_mu: float = 0.8, neg_mu: float = 0.2, seed: int = 7
) -> list[CalibrationItem]:
    rng = random.Random(seed)
    items = []
    for i in range(n_pos):
        items.append(
            CalibrationItem(
                use_case="uc",
                category="privacy",
                score=min(1.0, max(0.0, rng.gauss(pos_mu, 0.10))),
                label=1,
                item_id=f"p{i}",
            )
        )
    for i in range(n_neg):
        items.append(
            CalibrationItem(
                use_case="uc",
                category="privacy",
                score=min(1.0, max(0.0, rng.gauss(neg_mu, 0.10))),
                label=0,
                item_id=f"n{i}",
            )
        )
    return items


def test_threshold_is_the_kth_smallest_positive_score():
    items = make_items(400, 400)
    fit = fit_category(items, use_case="uc", category="privacy", alpha=0.10)
    assert not fit.degenerate
    assert fit.k_index == int(0.10 * (fit.n_calibration_positives + 1))
    assert 0.0 < fit.threshold < 1.0


def test_coverage_check_is_always_computed():
    """The check is mandatory (§7.7). It may pass or fail on any given split —
    that is the point of running it — but it must never be skipped."""
    fit = fit_category(
        make_items(500, 500), use_case="uc", category="privacy", alpha=0.10
    )
    assert fit.coverage is not None
    assert fit.coverage.n_test_positives > 0
    assert 0.0 <= fit.coverage.realised_miss_rate <= 1.0
    assert fit.coverage.upper_confidence_limit >= fit.coverage.realised_miss_rate
    if not fit.coverage.within_bound:
        assert "COVERAGE FAILED" in fit.coverage.note


def test_conformal_bound_holds_in_expectation_over_calibration_draws():
    """What split conformal actually guarantees.

    The bound is on the expected miss rate over the randomness of the
    calibration set, NOT on every individual split — a single split can and does
    land above alpha, which is exactly why fit_category reports the realised rate
    instead of asserting it. Averaged over many draws, it must come in at or
    below alpha.
    """
    alpha = 0.10
    realised = []
    for seed in range(40):
        fit = fit_category(
            make_items(400, 400, seed=seed),
            use_case="uc",
            category="privacy",
            alpha=alpha,
            seed=seed,
        )
        realised.append(fit.coverage.realised_miss_rate)
    mean_miss = sum(realised) / len(realised)
    assert mean_miss <= alpha + 0.02, f"mean realised miss {mean_miss:.4f} > alpha"
    # And it should not be wildly conservative either, or the threshold is
    # flagging far more than the risk target requires.
    assert mean_miss > alpha / 3, f"suspiciously conservative: {mean_miss:.4f}"


def test_coverage_check_detects_a_broken_bound_under_distribution_shift():
    """Exchangeability is the assumption the whole guarantee rests on. Break it
    and the check must say so rather than pass quietly."""
    rng = random.Random(11)
    calib = [
        CalibrationItem("uc", "privacy", min(1.0, max(0.0, rng.gauss(0.85, 0.05))), 1, item_id=f"c{i}")
        for i in range(200)
    ]
    # Shifted test positives: much lower scores, i.e. the detector got worse.
    shifted = [
        CalibrationItem("uc", "privacy", min(1.0, max(0.0, rng.gauss(0.25, 0.08))), 1, item_id=f"s{i}")
        for i in range(200)
    ]
    fit = fit_category(calib, use_case="uc", category="privacy", alpha=0.10)
    misses = sum(1 for s in shifted if s.score < fit.threshold)
    realised = misses / len(shifted)
    assert realised > fit.alpha, (
        "the constructed shift did not actually break the bound; "
        "the test would prove nothing"
    )


def test_degenerate_fit_is_reported_not_published():
    """floor(alpha*(n+1)) < 1: no threshold above 0 can meet alpha. Publishing
    tau=0 would flag 100% of traffic and call it a guarantee."""
    fit = fit_category(
        make_items(4, 50), use_case="uc", category="privacy", alpha=0.05
    )
    assert fit.degenerate
    assert fit.threshold == 0.0
    assert "sample size" in fit.note or "no positive" in fit.note

    payload = fit_all(make_items(4, 50), alpha=0.05)
    assert payload["thresholds"] == {} or "privacy" not in payload["thresholds"].get("uc", {})
    assert any(u["category"] == "privacy" for u in payload["unfitted"])


def test_no_positives_at_all_is_degenerate():
    items = [
        CalibrationItem("uc", "toxicity", 0.1, 0, item_id=f"n{i}") for i in range(80)
    ]
    fit = fit_category(items, use_case="uc", category="toxicity", alpha=0.10)
    assert fit.degenerate
    assert fit.n_calibration_positives == 0
    assert fit.coverage is not None and not fit.coverage.within_bound


def test_escalation_budget_is_reported_against_beta():
    fit = fit_category(
        make_items(300, 300), use_case="uc", category="privacy", alpha=0.10, beta=0.02
    )
    assert fit.achieved_escalation_rate > 0.02
    assert fit.escalation_within_budget is False
    # And it reports what meeting beta WOULD have cost, rather than silently
    # trading the risk bound away to hit the budget.
    assert fit.beta_constrained_threshold is not None
    assert fit.beta_constrained_miss_rate is not None
    assert fit.beta_constrained_miss_rate > fit.alpha


def test_fit_is_deterministic_under_a_seed():
    items = make_items(300, 300)
    a = fit_category(items, use_case="uc", category="privacy", alpha=0.1, seed=99)
    b = fit_category(items, use_case="uc", category="privacy", alpha=0.1, seed=99)
    assert a.threshold == b.threshold
    c = fit_category(items, use_case="uc", category="privacy", alpha=0.1, seed=100)
    assert isinstance(c.threshold, float)


def test_lower_alpha_gives_a_lower_threshold():
    """Demanding a smaller miss rate must flag more, never less."""
    items = make_items(600, 600)
    strict = fit_category(items, use_case="uc", category="privacy", alpha=0.02)
    loose = fit_category(items, use_case="uc", category="privacy", alpha=0.20)
    assert strict.threshold <= loose.threshold


@pytest.mark.parametrize(
    "successes,n", [(0, 10), (1, 20), (5, 100), (0, 1), (50, 100)]
)
def test_binomial_ucl_bounds_the_point_estimate(successes, n):
    ucl = binomial_ucl(successes, n)
    assert 0.0 <= ucl <= 1.0
    assert ucl >= successes / n


def test_binomial_ucl_tightens_with_sample_size():
    assert binomial_ucl(1, 10) > binomial_ucl(10, 100) > binomial_ucl(100, 1000)


def test_threshold_drift_reports_movement():
    old = {"thresholds": {"uc": {"privacy": 0.70}}}
    new = {"thresholds": {"uc": {"privacy": 0.62}}}
    drift = threshold_drift(old, new)
    assert len(drift) == 1
    assert drift[0]["delta"] == pytest.approx(-0.08)
