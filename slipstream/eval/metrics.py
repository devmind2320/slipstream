"""Metric computation for the three-arm comparison (§10).

Two rules from §9 shape this file:

* **Never report a single accuracy figure.** Every category gets an operating
  curve, and the chosen point is reported alongside it with the reason.
* **``flags_per_100_interactions`` per use case is a first-class metric**, because
  a system that catches everything by flagging everything has not solved the
  problem the brief describes.

Multi-label metrics are computed at both the per-category level and the
set-exact level, and they are reported separately — set-exact accuracy alone
hides which category the system got wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..contracts import Action, RiskCategory

BENIGN = "benign"
RISKY = "risky"


@dataclass
class ItemResult:
    item_id: str
    slice: str
    use_case: str
    arm: str
    band: str
    labels: list[str]
    predicted: list[str]
    action: str
    signal_mode: str = "proxy"

    ttft_ms: float = 0.0
    first_token_ms: float = 0.0
    total_latency_ms: float = 0.0
    generation_ms: float = 0.0
    guard_overhead_ms: float = 0.0

    masked_before_emission: bool = False
    entities_masked: list[str] = field(default_factory=list)
    leaked_chunks: int = 0
    abstained: bool = False
    expects_abstention: bool = False
    degraded: bool = False
    escalated: bool = False
    regenerated: bool = False
    # Detectors that overran their guard budget and abstained. Like degradation,
    # this is wall-clock dependent, so it makes the item's decision
    # non-reproducible.
    budget_timeouts: int = 0

    @property
    def load_dependent(self) -> bool:
        return self.degraded or self.budget_timeouts > 0

    guard_first_won: int = 0
    guard_first_total: int = 0

    tokens_in: int = 0
    tokens_out: int = 0
    model_used: str = ""
    routing_tier: str | None = None
    baseline_cost: float = 0.0
    actual_cost: float = 0.0
    cache_outcome: str = "miss"

    category_scores: dict[str, float] = field(default_factory=dict)

    @property
    def added_latency_ms(self) -> float:
        return max(0.0, self.total_latency_ms - self.generation_ms)

    @property
    def ttft_delta_ms(self) -> float:
        return max(0.0, self.ttft_ms - self.first_token_ms)

    @property
    def traffic(self) -> str:
        return RISKY if self.labels else BENIGN

    @property
    def flagged(self) -> bool:
        """Anything that was not silently allowed."""
        return self.action != Action.ALLOW.value

    @property
    def intervened(self) -> bool:
        """The subset that changed the response or pulled in a human.

        An ANNOTATE that says "no source was attached, so this was not checked"
        is an honest disclosure, not an alert. Counting it as one would overstate
        alert fatigue by the size of the unverifiable slice, so both numbers are
        reported and the difference is visible.
        """
        return self.action in (
            Action.MASK.value,
            Action.REGENERATE.value,
            Action.ESCALATE.value,
            Action.BLOCK.value,
        )


# ---------------------------------------------------------------------------
def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile. No numpy dependency in the report path."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _rate(num: int, den: int) -> float:
    return num / den if den else float("nan")


def latency_block(results: Iterable[ItemResult], traffic: str) -> dict:
    sub = [r for r in results if r.traffic == traffic]
    added = [r.added_latency_ms for r in sub]
    ttft = [r.ttft_ms for r in sub]
    delta = [r.ttft_delta_ms for r in sub]
    return {
        "n": len(sub),
        "added_latency_p50_ms": round(percentile(added, 0.50), 1),
        "added_latency_p95_ms": round(percentile(added, 0.95), 1),
        "ttft_p50_ms": round(percentile(ttft, 0.50), 1),
        "ttft_p95_ms": round(percentile(ttft, 0.95), 1),
        "ttft_delta_p50_ms": round(percentile(delta, 0.50), 1),
        "ttft_delta_p95_ms": round(percentile(delta, 0.95), 1),
    }


def pii_block(results: Sequence[ItemResult]) -> dict:
    """Interception is not the same as prevention.

    ``interception_rate``   — the system detected the outbound PII at all.
    ``masked_before_emission`` — it was withheld from the user. The gap between
    these two numbers IS the fixed_async arm's failure mode.
    """
    pii_items = [r for r in results if "privacy" in r.labels]
    intercepted = [r for r in pii_items if "privacy" in r.predicted]
    prevented = [
        r for r in pii_items if r.masked_before_emission and r.leaked_chunks == 0
    ]
    return {
        "n_pii_items": len(pii_items),
        "interception_rate": round(_rate(len(intercepted), len(pii_items)), 4),
        "masked_before_emission_rate": round(_rate(len(prevented), len(pii_items)), 4),
        "leaked_items": sum(1 for r in pii_items if r.leaked_chunks > 0),
    }


def category_block(results: Sequence[ItemResult], category: str) -> dict:
    tp = sum(1 for r in results if category in r.labels and category in r.predicted)
    fn = sum(1 for r in results if category in r.labels and category not in r.predicted)
    fp = sum(1 for r in results if category not in r.labels and category in r.predicted)
    tn = sum(
        1 for r in results if category not in r.labels and category not in r.predicted
    )
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else float("nan")
    )
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4) if precision == precision else None,
        "recall": round(recall, 4) if recall == recall else None,
        "f1": round(f1, 4) if f1 == f1 else None,
    }


def operating_curve(
    results: Sequence[ItemResult], category: str, thresholds: Sequence[float]
) -> list[dict]:
    """Precision/recall/flag-rate across candidate thresholds (§9).

    Reported instead of a single accuracy number so the chosen operating point
    can be seen in context rather than asserted.
    """
    curve: list[dict] = []
    scored = [r for r in results if category in r.category_scores]
    for t in thresholds:
        tp = sum(
            1 for r in scored if category in r.labels and r.category_scores[category] >= t
        )
        fp = sum(
            1
            for r in scored
            if category not in r.labels and r.category_scores[category] >= t
        )
        fn = sum(
            1 for r in scored if category in r.labels and r.category_scores[category] < t
        )
        flagged = tp + fp
        curve.append(
            {
                "threshold": round(t, 3),
                "precision": round(_rate(tp, tp + fp), 4),
                "recall": round(_rate(tp, tp + fn), 4),
                "flags_per_100": round(100.0 * _rate(flagged, len(scored)), 2),
            }
        )
    return curve


def abstention_block(results: Sequence[ItemResult]) -> dict:
    """Abstention is a first-class metric (§9), with a correctness split.

    Abstaining where no source exists is correct behaviour. Abstaining where a
    source WAS attached is a detector failure wearing a polite face.
    """
    should = [r for r in results if r.expects_abstention]
    should_not = [r for r in results if not r.expects_abstention]
    return {
        "abstention_rate": round(_rate(sum(1 for r in results if r.abstained), len(results)), 4),
        "n_expected_abstain": len(should),
        "correct_abstention_rate": round(
            _rate(sum(1 for r in should if r.abstained), len(should)), 4
        ),
        "spurious_abstention_rate": round(
            _rate(sum(1 for r in should_not if r.abstained), len(should_not)), 4
        ),
    }


def set_exact_accuracy(results: Sequence[ItemResult]) -> dict:
    exact = sum(1 for r in results if set(r.labels) == set(r.predicted))
    overlap_items = [r for r in results if len(r.labels) > 1]
    overlap_exact = sum(
        1 for r in overlap_items if set(r.labels) == set(r.predicted)
    )
    overlap_partial = sum(
        1 for r in overlap_items if set(r.labels) & set(r.predicted)
    )
    jaccard_sum = 0.0
    for r in results:
        a, b = set(r.labels), set(r.predicted)
        jaccard_sum += 1.0 if not a and not b else len(a & b) / len(a | b)
    return {
        "set_exact_accuracy": round(_rate(exact, len(results)), 4),
        "mean_jaccard": round(jaccard_sum / len(results), 4) if results else None,
        "n_multilabel_items": len(overlap_items),
        "multilabel_set_exact": round(_rate(overlap_exact, len(overlap_items)), 4),
        "multilabel_any_overlap": round(_rate(overlap_partial, len(overlap_items)), 4),
    }


def flags_per_100(results: Sequence[ItemResult], attr: str = "flagged") -> dict[str, float]:
    """Alert-fatigue proxy, per use case (§9)."""
    by_uc: dict[str, list[ItemResult]] = {}
    for r in results:
        by_uc.setdefault(r.use_case, []).append(r)
    out = {
        uc: round(100.0 * _rate(sum(1 for r in rs if getattr(r, attr)), len(rs)), 2)
        for uc, rs in sorted(by_uc.items())
    }
    out["all"] = round(
        100.0 * _rate(sum(1 for r in results if getattr(r, attr)), len(results)), 2
    )
    return out


def escalation_block(results: Sequence[ItemResult], budgets: dict[str, float]) -> dict:
    by_uc: dict[str, list[ItemResult]] = {}
    for r in results:
        by_uc.setdefault(r.use_case, []).append(r)
    out: dict[str, dict] = {}
    for uc, rs in sorted(by_uc.items()):
        rate = _rate(sum(1 for r in rs if r.escalated), len(rs))
        benign = [r for r in rs if r.traffic == BENIGN]
        benign_rate = _rate(sum(1 for r in benign if r.escalated), len(benign))
        budget = budgets.get(uc, float("nan"))
        out[uc] = {
            "escalation_rate": round(rate, 4),
            "escalation_rate_benign": round(benign_rate, 4),
            "n_benign": len(benign),
            "risky_share_of_corpus": round(
                _rate(len(rs) - len(benign), len(rs)), 4
            ),
            "budget": budget,
            # The pack's beta is a target share of *production* traffic. This
            # corpus is deliberately enriched with risky items, so the overall
            # rate is not comparable to it; the benign-traffic rate is the one
            # that speaks to alert fatigue on real volume. Both are reported and
            # the comparison is made against the benign figure, with the risky
            # share of the corpus alongside it so the reader can judge.
            "within_budget": (
                bool(benign_rate <= budget) if budget == budget else None
            ),
        }
    benign_all = [r for r in results if r.traffic == BENIGN]
    out["all"] = {
        "escalation_rate": round(
            _rate(sum(1 for r in results if r.escalated), len(results)), 4
        ),
        "escalation_rate_benign": round(
            _rate(sum(1 for r in benign_all if r.escalated), len(benign_all)), 4
        ),
        "n_benign": len(benign_all),
        "risky_share_of_corpus": round(
            _rate(len(results) - len(benign_all), len(results)), 4
        ),
        "budget": None,
        "within_budget": None,
    }
    return out


def cost_block(results: Sequence[ItemResult]) -> dict:
    """Net token cost delta (routing + cache - guard).

    The guard term is **zero tokens** in this build: every detector is a local
    CPU model, not an LLM judge. Stated explicitly rather than left implied,
    because a design that used an LLM as a guard would have a very different
    number here and the comparison would not carry over.
    """
    baseline = sum(r.baseline_cost for r in results)
    actual = sum(r.actual_cost for r in results)
    cheap = sum(1 for r in results if r.model_used.endswith("cheap"))
    cache_hits = sum(1 for r in results if r.cache_outcome in ("exact", "semantic"))
    gated = sum(1 for r in results if r.cache_outcome == "gated_miss")
    return {
        "baseline_cost_usd": round(baseline, 6),
        "actual_cost_usd": round(actual, 6),
        "net_delta_usd": round(actual - baseline, 6),
        "net_delta_pct": (
            round(100.0 * (actual - baseline) / baseline, 2) if baseline else None
        ),
        "guard_token_cost_usd": 0.0,
        "routed_cheap": cheap,
        "cache_hits": cache_hits,
        "cache_gated_misses": gated,
    }


def scheduler_block(results: Sequence[ItemResult]) -> dict:
    won = sum(r.guard_first_won for r in results)
    total = sum(r.guard_first_total for r in results)
    by_band: dict[str, dict] = {}
    for band in sorted({r.band for r in results}):
        sub = [r for r in results if r.band == band]
        w = sum(r.guard_first_won for r in sub)
        t = sum(r.guard_first_total for r in sub)
        by_band[band] = {
            "chunks_with_a_successor": t,
            "pct_guard_finished_first": round(_rate(w, t), 4),
        }
    return {
        "chunks_with_a_successor": total,
        "pct_chunks_guard_finished_first": round(_rate(won, total), 4),
        "by_band": by_band,
        "degraded_mode_rate": round(
            _rate(sum(1 for r in results if r.degraded), len(results)), 4
        ),
        "budget_timeout_rate": round(
            _rate(sum(1 for r in results if r.budget_timeouts), len(results)), 4
        ),
        "load_dependent_rate": round(
            _rate(sum(1 for r in results if r.load_dependent), len(results)), 4
        ),
        "regeneration_rate": round(
            _rate(sum(1 for r in results if r.regenerated), len(results)), 4
        ),
        "mean_guard_overhead_ms": round(
            sum(r.guard_overhead_ms for r in results) / len(results), 2
        )
        if results
        else None,
    }


def signal_mode_block(results: Sequence[ItemResult]) -> dict:
    """§7.4 requires performance to be reported separately per signal set."""
    out: dict[str, dict] = {}
    for mode in sorted({r.signal_mode for r in results}):
        sub = [r for r in results if r.signal_mode == mode]
        out[mode] = {
            "n": len(sub),
            "share": round(_rate(len(sub), len(results)), 4),
            "hallucination": category_block(sub, "hallucination"),
        }
    return out


CURVE_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]


def sanitize(obj):
    """NaN/Infinity -> None, recursively.

    ``json.dumps`` emits bare ``NaN`` by default, which is not valid JSON and is
    rejected by strict parsers. results.json is the deliverable, so it has to be
    a file anyone can open. "Not computable" is expressed as null, and the report
    renders it as `n/a`.
    """
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def compute_arm_metrics(
    results: Sequence[ItemResult], budgets: dict[str, float]
) -> dict:
    return sanitize({
        "n_items": len(results),
        "latency_benign": latency_block(results, BENIGN),
        "latency_risky": latency_block(results, RISKY),
        "scheduler": scheduler_block(results),
        "pii": pii_block(results),
        "per_category": {
            c.value: category_block(results, c.value) for c in RiskCategory
        },
        "abstention": abstention_block(results),
        "multilabel": set_exact_accuracy(results),
        "flags_per_100_interactions": flags_per_100(results, "flagged"),
        "interventions_per_100_interactions": flags_per_100(results, "intervened"),
        "escalation": escalation_block(results, budgets),
        "cost": cost_block(results),
        "signal_modes": signal_mode_block(results),
        "operating_curves": {
            c: operating_curve(results, c, CURVE_THRESHOLDS)
            for c in ("privacy", "hallucination", "injection", "bias")
        },
        "action_distribution": {
            a.value: sum(1 for r in results if r.action == a.value) for a in Action
        },
    })
