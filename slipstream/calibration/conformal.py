"""Split-conformal risk control (§7.7).

This is what turns a tuned number into a stated guarantee, and it is the part of
the pitch that is actually a claim rather than an engineering choice.

**The construction.** For a risk category, the nonconformity score is the
detector's score. Take the labelled *positive* items (things that genuinely are
violations) in the calibration split, sort their scores ascending, and set

    k = floor(alpha * (n + 1)),   tau = s_(k)      (the k-th smallest score)

Under exchangeability of calibration and test positives, a fresh positive item
scores below tau with probability at most alpha. Scoring below tau is exactly a
miss. So: **expected miss rate <= alpha**, distribution-free, no assumption about
the detector being well-calibrated or even good.

When k < 1 the bound cannot be met by any threshold at this sample size; we set
tau = 0.0 (flag everything) and say so in the fit record rather than quietly
picking something plausible.

**What the guarantee does not cover**, stated here and in the README because it
is the honest limit: exchangeability. Calibration data and production traffic
must be drawn from the same distribution. Production drift — a new product line,
a new attacker, a new locale — violates it, and the bound goes with it. This is
why feedback/capture.py exists and why `slipstream recalibrate` reports drift.

**The coverage check is mandatory.** After fitting on one split we measure the
realised miss rate on a disjoint split and report whether it lands inside the
bound, with a binomial upper confidence limit. A calibration that fails its own
coverage check and admits it is a stronger result than one that never tested.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .. import settings
from ..contracts import RiskCategory


@dataclass(frozen=True)
class CalibrationItem:
    use_case: str
    category: str
    score: float
    label: int  # 1 = genuine violation of this category, 0 = not
    source: str = "eval"
    item_id: str = ""

    @staticmethod
    def from_dict(d: dict) -> "CalibrationItem":
        return CalibrationItem(
            use_case=d["use_case"],
            category=d["category"],
            score=float(d["score"]),
            label=int(d["label"]),
            source=d.get("source", "eval"),
            item_id=d.get("item_id", ""),
        )


@dataclass
class CoverageCheck:
    n_test_positives: int
    realised_miss_rate: float
    alpha: float
    upper_confidence_limit: float  # 95% binomial UCL on the miss rate
    within_bound: bool
    note: str = ""


@dataclass
class CategoryFit:
    use_case: str
    category: str
    alpha: float
    beta: float
    threshold: float
    n_calibration_positives: int
    n_calibration_total: int
    k_index: int
    achieved_escalation_rate: float
    escalation_within_budget: bool
    beta_constrained_threshold: float | None = None
    beta_constrained_miss_rate: float | None = None
    coverage: CoverageCheck | None = None
    degenerate: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def binomial_ucl(successes: int, n: int, confidence: float = 0.95) -> float:
    """One-sided upper confidence limit on a proportion (Clopper-Pearson).

    Implemented by bisection on the regularised incomplete beta so we do not pull
    in scipy for one function.
    """
    if n == 0:
        return 1.0
    if successes >= n:
        return 1.0
    alpha = 1.0 - confidence

    def binom_cdf(p: float) -> float:
        # P(X <= successes) for X ~ Bin(n, p)
        total = 0.0
        for i in range(successes + 1):
            total += math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
        return total

    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if binom_cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def _split(
    items: list[CalibrationItem], seed: int, calib_frac: float = 0.5
) -> tuple[list[CalibrationItem], list[CalibrationItem]]:
    """Stratified by label so both halves contain positives."""
    rng = random.Random(seed)
    pos = [i for i in items if i.label == 1]
    neg = [i for i in items if i.label == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    p_cut = int(len(pos) * calib_frac)
    n_cut = int(len(neg) * calib_frac)
    return (pos[:p_cut] + neg[:n_cut], pos[p_cut:] + neg[n_cut:])


def fit_category(
    items: list[CalibrationItem],
    *,
    use_case: str,
    category: str,
    alpha: float = 0.10,
    beta: float = 0.05,
    seed: int = settings.DEFAULT_SEED,
) -> CategoryFit:
    calib, test = _split(items, seed)
    cal_pos = sorted(i.score for i in calib if i.label == 1)
    n = len(cal_pos)

    k = math.floor(alpha * (n + 1))
    degenerate = False
    note = ""
    if n == 0:
        threshold = 0.0
        degenerate = True
        note = "no positive calibration examples; threshold pinned to 0.0 (flag all)"
    elif k < 1:
        threshold = 0.0
        degenerate = True
        note = (
            f"floor(alpha*(n+1)) = {k} < 1 with n={n}: no threshold above 0 can "
            f"meet alpha={alpha} at this sample size"
        )
    else:
        threshold = float(cal_pos[k - 1])

    # Escalation rate = share of ALL calibration traffic at or above tau. This
    # is the alert-fatigue side of the trade and it is reported against beta,
    # not silently traded away.
    flagged = sum(1 for i in calib if i.score >= threshold)
    esc_rate = flagged / len(calib) if calib else 0.0

    beta_thr: float | None = None
    beta_miss: float | None = None
    if calib and esc_rate > beta:
        all_scores = sorted((i.score for i in calib), reverse=True)
        take = max(1, int(round(beta * len(all_scores))))
        beta_thr = float(all_scores[min(take, len(all_scores)) - 1])
        beta_miss = (
            sum(1 for s in cal_pos if s < beta_thr) / len(cal_pos) if cal_pos else 1.0
        )

    # -- mandatory empirical coverage check on the held-out split ----------
    test_pos = [i.score for i in test if i.label == 1]
    misses = sum(1 for s in test_pos if s < threshold)
    realised = misses / len(test_pos) if test_pos else 0.0
    ucl = binomial_ucl(misses, len(test_pos))
    within = realised <= alpha
    cov_note = ""
    if not test_pos:
        cov_note = "no positive test examples; coverage NOT verified"
        within = False
    elif not within:
        cov_note = (
            f"COVERAGE FAILED: realised miss rate {realised:.3f} exceeds "
            f"alpha={alpha:.3f} on {len(test_pos)} held-out positives"
        )
    elif ucl > alpha:
        cov_note = (
            f"realised {realised:.3f} <= alpha but the 95% UCL is {ucl:.3f}; "
            f"sample size {len(test_pos)} is too small to confirm the bound"
        )

    return CategoryFit(
        use_case=use_case,
        category=category,
        alpha=alpha,
        beta=beta,
        threshold=round(threshold, 4),
        n_calibration_positives=n,
        n_calibration_total=len(calib),
        k_index=k,
        achieved_escalation_rate=round(esc_rate, 4),
        escalation_within_budget=esc_rate <= beta,
        beta_constrained_threshold=(round(beta_thr, 4) if beta_thr is not None else None),
        beta_constrained_miss_rate=(
            round(beta_miss, 4) if beta_miss is not None else None
        ),
        coverage=CoverageCheck(
            n_test_positives=len(test_pos),
            realised_miss_rate=round(realised, 4),
            alpha=alpha,
            upper_confidence_limit=round(ucl, 4),
            within_bound=within,
            note=cov_note,
        ),
        degenerate=degenerate,
        note=note,
    )


def fit_all(
    items: list[CalibrationItem],
    *,
    alpha: float = 0.10,
    beta_by_use_case: dict[str, float] | None = None,
    seed: int = settings.DEFAULT_SEED,
) -> dict:
    """Fit per use case and per category (§7.7)."""
    beta_by_use_case = beta_by_use_case or {}
    grouped: dict[tuple[str, str], list[CalibrationItem]] = {}
    for i in items:
        grouped.setdefault((i.use_case, i.category), []).append(i)

    fits: list[CategoryFit] = []
    for (use_case, category), group in sorted(grouped.items()):
        fits.append(
            fit_category(
                group,
                use_case=use_case,
                category=category,
                alpha=alpha,
                beta=beta_by_use_case.get(use_case, 0.05),
                seed=seed,
            )
        )

    # A degenerate fit means "no threshold above 0 can meet alpha at this sample
    # size". Publishing tau = 0.0 as a *usable* threshold would make the detector
    # fire on every input — technically satisfying the bound by flagging all
    # traffic, and destroying the system. Those pairs are reported as unfitted
    # and the policy pack's configured default stays in force.
    thresholds: dict[str, dict[str, float]] = {}
    unfitted: list[dict] = []
    for f in fits:
        if f.degenerate:
            unfitted.append(
                {"use_case": f.use_case, "category": f.category, "reason": f.note}
            )
            continue
        thresholds.setdefault(f.use_case, {})[f.category] = f.threshold

    return {
        "fit_date": datetime.now(timezone.utc).isoformat(),
        "split_seed": seed,
        "alpha": alpha,
        "n_items": len(items),
        "thresholds": thresholds,
        "unfitted": unfitted,
        "fits": [f.to_dict() for f in fits],
        "coverage_failures": [
            f"{f.use_case}/{f.category}"
            for f in fits
            if f.coverage and not f.coverage.within_bound and not f.degenerate
        ],
    }


def save_thresholds(payload: dict, path: Path | None = None) -> Path:
    path = Path(path or settings.THRESHOLDS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_thresholds(path: Path | None = None) -> dict:
    path = Path(path or settings.THRESHOLDS_PATH)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_calibration_set(path: Path | None = None) -> list[CalibrationItem]:
    path = Path(path or settings.CALIBRATION_SET_PATH)
    if not path.exists():
        return []
    out: list[CalibrationItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(CalibrationItem.from_dict(json.loads(line)))
    return out


def append_calibration_items(
    items: list[CalibrationItem], path: Path | None = None
) -> Path:
    path = Path(path or settings.CALIBRATION_SET_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for i in items:
            fh.write(json.dumps(asdict(i), sort_keys=True) + "\n")
    return path


def threshold_drift(old: dict, new: dict) -> list[dict]:
    """Per-category threshold movement between two fits. Drives `recalibrate`."""
    out: list[dict] = []
    old_t = old.get("thresholds", {})
    new_t = new.get("thresholds", {})
    for use_case in sorted(set(old_t) | set(new_t)):
        for cat in sorted(set(old_t.get(use_case, {})) | set(new_t.get(use_case, {}))):
            o = old_t.get(use_case, {}).get(cat)
            n = new_t.get(use_case, {}).get(cat)
            out.append(
                {
                    "use_case": use_case,
                    "category": cat,
                    "old": o,
                    "new": n,
                    "delta": (round(n - o, 4) if (o is not None and n is not None) else None),
                }
            )
    return out


CATEGORY_NAMES = tuple(c.value for c in RiskCategory)
