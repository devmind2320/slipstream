"""CLI: fit conformal thresholds from a labelled calibration set.

    python -m slipstream.calibration.fit --alpha 0.10
    python -m slipstream.calibration.fit --input data/calibration_set.jsonl --report

Writes calibration/thresholds.json (fit date, sample size, split seed) and prints
the coverage check for every use-case/category pair — including the failures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import settings
from ..policy.loader import get_loader
from .conformal import (
    fit_all,
    load_calibration_set,
    load_thresholds,
    save_thresholds,
    threshold_drift,
)


def _betas() -> dict[str, float]:
    loader = get_loader()
    out: dict[str, float] = {}
    for use_case in loader.available():
        try:
            out[use_case] = loader.load(use_case).escalation.budget
        except Exception:  # noqa: BLE001
            continue
    return out


def render_report(payload: dict, drift: list[dict] | None = None) -> str:
    lines: list[str] = []
    lines.append("# Conformal calibration report")
    lines.append("")
    lines.append(f"- fit date: `{payload['fit_date']}`")
    lines.append(f"- split seed: `{payload['split_seed']}`")
    lines.append(f"- target risk alpha: `{payload['alpha']}`")
    lines.append(f"- calibration items: `{payload['n_items']}`")
    lines.append("")
    lines.append(
        "| use case | category | tau | n pos | escalation vs budget | "
        "realised miss | alpha | 95% UCL | coverage |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for f in payload["fits"]:
        cov = f.get("coverage") or {}
        if f.get("degenerate"):
            lines.append(
                f"| {f['use_case']} | {f['category']} | — | "
                f"{f['n_calibration_positives']} | n/a | n/a | "
                f"{f['alpha']:.2f} | n/a | UNFITTED |"
            )
            continue
        verdict = "PASS" if cov.get("within_bound") else "**FAIL**"
        lines.append(
            f"| {f['use_case']} | {f['category']} | {f['threshold']:.3f} | "
            f"{f['n_calibration_positives']} | "
            f"{f['achieved_escalation_rate']:.3f} / {f['beta']:.3f}"
            f"{'' if f['escalation_within_budget'] else ' (!)'} | "
            f"{cov.get('realised_miss_rate', float('nan')):.3f} | {f['alpha']:.2f} | "
            f"{cov.get('upper_confidence_limit', float('nan')):.3f} | {verdict} |"
        )
    lines.append("")

    notes = [
        f"- `{f['use_case']}/{f['category']}`: {f['note'] or (f.get('coverage') or {}).get('note','')}"
        for f in payload["fits"]
        if f.get("note") or ((f.get("coverage") or {}).get("note"))
    ]
    if notes:
        lines.append("## Notes and failures")
        lines.append("")
        lines.extend(notes)
        lines.append("")

    if payload["coverage_failures"]:
        lines.append(
            f"**{len(payload['coverage_failures'])} coverage check(s) failed: "
            f"{', '.join(payload['coverage_failures'])}.** The conformal bound is "
            "not honoured for these pairs at this sample size. Reported rather "
            "than suppressed, per spec §7.7."
        )
    else:
        lines.append(
            "All coverage checks passed: the realised miss rate on the held-out "
            "split fell within the stated bound for every use-case/category pair."
        )
    lines.append("")

    if drift:
        lines.append("## Threshold drift vs previous fit")
        lines.append("")
        lines.append("| use case | category | old | new | delta |")
        lines.append("|---|---|---|---|---|")
        for d in drift:
            lines.append(
                f"| {d['use_case']} | {d['category']} | {d['old']} | {d['new']} | "
                f"{d['delta']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit split-conformal thresholds")
    ap.add_argument("--input", type=Path, default=settings.CALIBRATION_SET_PATH)
    ap.add_argument("--output", type=Path, default=settings.THRESHOLDS_PATH)
    ap.add_argument("--alpha", type=float, default=0.10, help="target miss rate")
    ap.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
    ap.add_argument("--report", action="store_true", help="print a markdown report")
    ap.add_argument("--report-out", type=Path, default=None)
    args = ap.parse_args(argv)

    items = load_calibration_set(args.input)
    if not items:
        print(
            f"no calibration items at {args.input}. Run `make eval` first — the "
            f"harness writes the labelled set."
        )
        return 1

    previous = load_thresholds(args.output)
    payload = fit_all(items, alpha=args.alpha, beta_by_use_case=_betas(), seed=args.seed)
    drift = threshold_drift(previous, payload) if previous else None
    save_thresholds(payload, args.output)

    print(f"fitted {len(payload['fits'])} use-case/category pairs -> {args.output}")
    if payload["coverage_failures"]:
        print("COVERAGE FAILURES: " + ", ".join(payload["coverage_failures"]))

    if args.report or args.report_out:
        report = render_report(payload, drift)
        if args.report_out:
            args.report_out.parent.mkdir(parents=True, exist_ok=True)
            args.report_out.write_text(report, encoding="utf-8")
            print(f"report -> {args.report_out}")
        if args.report:
            print()
            print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
