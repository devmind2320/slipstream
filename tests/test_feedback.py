"""Feedback loop: human overrides -> calibration set -> refit -> reported drift.

§7.11's claim is that the loop is *demonstrable*, not asserted. This file
demonstrates it: capture reviewer labels, refit, and assert the thresholds
actually moved and that the movement is reported.
"""

from __future__ import annotations

import json

import pytest

from slipstream.calibration.conformal import (
    CalibrationItem,
    append_calibration_items,
    fit_all,
    load_calibration_set,
    save_thresholds,
)
from slipstream.contracts import (
    Action,
    Band,
    DecisionRecord,
    RiskCategory,
    RiskVector,
    StreamPolicy,
    Verdict,
)
from slipstream.feedback.capture import FeedbackCapture, Override, recalibrate, render_drift
from slipstream.ledger.store import Ledger


def make_record(score: float, rid: str) -> DecisionRecord:
    return DecisionRecord(
        record_id=rid,
        use_case="internal_knowledge",
        risk_vector=RiskVector(scores={RiskCategory.PRIVACY: score}),
        policy=StreamPolicy(band=Band.HIGH),
        action=Action.ESCALATE,
        verdicts=[
            Verdict(
                detector="pii",
                category=RiskCategory.PRIVACY,
                score=score,
                latency_ms=2.0,
                evidence="entities=EMAIL",
            )
        ],
        escalated=True,
    )


def test_capture_writes_one_labelled_example_per_override(tmp_path):
    path = tmp_path / "calibration_set.jsonl"
    ledger = Ledger(tmp_path / "l.sqlite3")
    capture = FeedbackCapture(ledger=ledger, calibration_path=path)

    rec = make_record(0.71, "a" * 32)
    ledger.append(rec)
    item = capture.capture(
        rec,
        Override(
            record_id=rec.record_id,
            category=RiskCategory.PRIVACY,
            reviewer_label=1,
            reviewer="alex",
            note="genuine leak, agent should not have said it",
        ),
    )

    assert item.score == pytest.approx(0.71)
    assert item.label == 1
    assert item.source == "human:alex"
    assert item.item_id == rec.record_id

    written = load_calibration_set(path)
    assert len(written) == 1 and written[0].source == "human:alex"

    # The ledger records the override WITHOUT mutating the decision (INV-6).
    fb = ledger.feedback_rows()
    assert len(fb) == 1 and fb[0]["reviewer"] == "alex"
    assert ledger.get(rec.record_id)["action"] == "escalate"
    ledger.close()


def test_reviewer_disagreement_moves_the_threshold_and_reports_drift(tmp_path):
    """The loop end to end: reviewers say the low-scoring flags were real, and
    the refit lowers the threshold to catch them."""
    cal_path = tmp_path / "calibration_set.jsonl"
    thr_path = tmp_path / "thresholds.json"

    # Baseline: positives cluster high, so tau lands high.
    baseline = [
        CalibrationItem("internal_knowledge", "privacy", 0.80 + 0.001 * i, 1, item_id=f"p{i}")
        for i in range(120)
    ] + [
        CalibrationItem("internal_knowledge", "privacy", 0.10 + 0.001 * i, 0, item_id=f"n{i}")
        for i in range(120)
    ]
    append_calibration_items(baseline, cal_path)
    first = fit_all(baseline, alpha=0.10, seed=7)
    save_thresholds(first, thr_path)
    tau_before = first["thresholds"]["internal_knowledge"]["privacy"]

    # Reviewers overturn a batch of low-scoring cases as genuine violations.
    ledger = Ledger(tmp_path / "l.sqlite3")
    capture = FeedbackCapture(ledger=ledger, calibration_path=cal_path)
    for i in range(60):
        rec = make_record(0.30 + 0.002 * i, f"{i:032x}")
        ledger.append(rec)
        capture.capture(
            rec,
            Override(
                record_id=rec.record_id,
                category=RiskCategory.PRIVACY,
                reviewer_label=1,
                reviewer="reviewer",
            ),
        )

    result = recalibrate(
        calibration_path=cal_path, thresholds_path=thr_path, alpha=0.10, seed=7
    )
    assert "error" not in result
    assert result["n_human_labels"] == 60

    tau_after = json.loads(thr_path.read_text(encoding="utf-8"))["thresholds"][
        "internal_knowledge"
    ]["privacy"]
    assert tau_after < tau_before, (
        f"reviewers said the low-scoring cases were real, but tau did not drop "
        f"({tau_before} -> {tau_after})"
    )

    moved = [d for d in result["drift"] if d["delta"] not in (None, 0.0)]
    assert moved, "drift was not reported"
    assert result["n_moved"] >= 1

    rendered = render_drift(result)
    assert "internal_knowledge/privacy" in rendered
    assert "down" in rendered
    ledger.close()


def test_recalibrate_with_no_data_reports_an_error(tmp_path):
    result = recalibrate(
        calibration_path=tmp_path / "missing.jsonl",
        thresholds_path=tmp_path / "t.json",
    )
    assert "error" in result
    assert "no calibration items" in render_drift(result)


def test_capture_records_zero_score_when_the_category_abstained(tmp_path):
    """An abstention carries no score to learn from; the reviewer's label is
    still captured, at score 0.0, rather than being invented."""
    rec = make_record(0.0, "b" * 32).model_copy(
        update={
            "verdicts": [
                Verdict(
                    detector="groundedness",
                    category=RiskCategory.HALLUCINATION,
                    score=0.0,
                    latency_ms=1.0,
                    abstained=True,
                    evidence="no_grounding_source",
                )
            ]
        }
    )
    capture = FeedbackCapture(calibration_path=tmp_path / "c.jsonl")
    item = capture.capture(
        rec,
        Override(
            record_id=rec.record_id,
            category=RiskCategory.HALLUCINATION,
            reviewer_label=0,
        ),
    )
    assert item.score == 0.0 and item.label == 0
