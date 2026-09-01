"""Human override capture and recalibration (§7.11).

Escalated items go to a human. What the human decides is the only ground truth
this system ever sees, so it goes straight back into the calibration set with the
reviewer's label. ``slipstream recalibrate`` then refits and **reports threshold
drift**, which is what makes the feedback loop demonstrable instead of asserted.

The override is also written to the ledger's feedback table, keyed to the
decision record it corrects — the decision record itself is never mutated
(INV-6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .. import settings
from ..calibration.conformal import (
    CalibrationItem,
    append_calibration_items,
    fit_all,
    load_calibration_set,
    load_thresholds,
    save_thresholds,
    threshold_drift,
)
from ..contracts import DecisionRecord, RiskCategory
from ..ledger.store import Ledger


@dataclass
class Override:
    record_id: str
    category: RiskCategory
    reviewer_label: int  # 1 = the reviewer agrees this is a violation
    reviewer: str = "human"
    note: str = ""


class FeedbackCapture:
    def __init__(
        self,
        ledger: Ledger | None = None,
        calibration_path: Path | None = None,
    ) -> None:
        self.ledger = ledger
        self.calibration_path = Path(calibration_path or settings.CALIBRATION_SET_PATH)

    def capture(self, record: DecisionRecord, override: Override) -> CalibrationItem:
        """Turn one human decision into one labelled calibration example."""
        score = 0.0
        for v in record.verdicts:
            if v.category is override.category and not v.abstained:
                score = max(score, v.score)

        item = CalibrationItem(
            use_case=record.use_case,
            category=override.category.value,
            score=score,
            label=override.reviewer_label,
            source=f"human:{override.reviewer}",
            item_id=record.record_id,
        )
        append_calibration_items([item], self.calibration_path)

        if self.ledger is not None:
            self.ledger.add_feedback(
                record.record_id,
                label=str(override.reviewer_label),
                reviewer=override.reviewer,
                category=override.category.value,
                note=override.note,
            )
        return item

    def capture_many(
        self, pairs: list[tuple[DecisionRecord, Override]]
    ) -> list[CalibrationItem]:
        return [self.capture(r, o) for r, o in pairs]


def recalibrate(
    *,
    calibration_path: Path | None = None,
    thresholds_path: Path | None = None,
    alpha: float = 0.10,
    seed: int = settings.DEFAULT_SEED,
) -> dict:
    """Refit from the current calibration set and report what moved."""
    from ..policy.loader import get_loader

    calibration_path = Path(calibration_path or settings.CALIBRATION_SET_PATH)
    thresholds_path = Path(thresholds_path or settings.THRESHOLDS_PATH)

    items = load_calibration_set(calibration_path)
    if not items:
        return {"error": f"no calibration items at {calibration_path}"}

    previous = load_thresholds(thresholds_path)
    loader = get_loader()
    betas = {}
    for uc in loader.available():
        try:
            betas[uc] = loader.load(uc).escalation.budget
        except Exception:  # noqa: BLE001
            pass

    payload = fit_all(items, alpha=alpha, beta_by_use_case=betas, seed=seed)
    drift = threshold_drift(previous, payload) if previous else []
    save_thresholds(payload, thresholds_path)

    human_items = sum(1 for i in items if i.source.startswith("human:"))
    moved = [d for d in drift if d["delta"] not in (None, 0.0)]
    return {
        "n_items": len(items),
        "n_human_labels": human_items,
        "drift": drift,
        "n_moved": len(moved),
        "coverage_failures": payload["coverage_failures"],
        "thresholds_path": str(thresholds_path),
    }


def render_drift(result: dict) -> str:
    if "error" in result:
        return result["error"]
    lines = [
        f"recalibrated on {result['n_items']} items "
        f"({result['n_human_labels']} from human reviewers)",
        f"{result['n_moved']} threshold(s) moved:",
    ]
    for d in result["drift"]:
        if d["delta"] in (None, 0.0):
            continue
        arrow = "up" if d["delta"] > 0 else "down"
        lines.append(
            f"  {d['use_case']}/{d['category']}: {d['old']} -> {d['new']} "
            f"({arrow} {abs(d['delta']):.4f})"
        )
    if result["coverage_failures"]:
        lines.append("coverage failures: " + ", ".join(result["coverage_failures"]))
    return "\n".join(lines)
