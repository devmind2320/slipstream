"""Slipstream CLI.

    python -m slipstream ledger verify
    python -m slipstream recalibrate
    python -m slipstream fit --alpha 0.10
    python -m slipstream data --seed 1729
    python -m slipstream eval --seed 1729
    python -m slipstream demo --scenario compensation
    python -m slipstream serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import settings


def cmd_ledger(args) -> int:
    from .ledger.store import Ledger, verify_chain

    ledger = Ledger(args.path)
    if args.ledger_cmd == "verify":
        brk = verify_chain(ledger)
        if brk is None:
            print(f"ok: {len(ledger)} record(s), hash chain intact")
            return 0
        print(
            f"CHAIN BREAK at seq {brk.seq} (record {brk.record_id}): {brk.kind}\n"
            f"  expected {brk.expected}\n  found    {brk.found}"
        )
        return 2
    if args.ledger_cmd == "show":
        for r in ledger.recent(args.limit):
            print(
                f"{r['seq']:>5}  {r['record_id'][:12]}  {r['arm']:<12} "
                f"{r['use_case']:<20} {r['band']:<7} {r['action']:<10} "
                f"{r['record_hash'][:16]}"
            )
        return 0
    return 1


def cmd_recalibrate(args) -> int:
    from .feedback.capture import recalibrate, render_drift

    result = recalibrate(alpha=args.alpha, seed=args.seed)
    print(render_drift(result))
    return 0 if "error" not in result else 1


def cmd_fit(args) -> int:
    from .calibration.fit import main as fit_main

    argv = ["--alpha", str(args.alpha), "--seed", str(args.seed)]
    if args.report:
        argv.append("--report")
    return fit_main(argv)


def cmd_data(args) -> int:
    from .eval.datasets.generate import main as gen_main

    return gen_main(["--seed", str(args.seed), "--scale", str(args.scale)])


def cmd_eval(args) -> int:
    from .eval.harness import main as harness_main

    argv = [
        "--seed", str(args.seed),
        "--scale", str(args.scale),
        "--alpha", str(args.alpha),
        "--concurrency", str(args.concurrency),
    ]
    if args.dump_items:
        argv.append("--dump-items")
    return harness_main(argv)


def cmd_demo(args) -> int:
    from .demo.run import main as demo_main

    return demo_main(["--scenario", args.scenario])


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run(
        "slipstream.gateway.app:app", host=args.host, port=args.port, log_level="warning"
    )
    return 0


def cmd_info(args) -> int:
    from .guards.registry import default_registry
    from .policy.loader import get_loader

    reg = default_registry()
    print(json.dumps({
        "version": __import__("slipstream").__version__,
        "backends": settings.backend_summary(),
        "detectors": reg.names,
        "registry_fingerprint": reg.fingerprint(),
        "packs": get_loader().available(),
        "ledger": str(settings.LEDGER_PATH),
        "thresholds": str(settings.THRESHOLDS_PATH),
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="slipstream", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ledger", help="inspect and verify the decision ledger")
    p.add_argument("ledger_cmd", choices=["verify", "show"])
    p.add_argument("--path", type=Path, default=settings.LEDGER_PATH)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=cmd_ledger)

    p = sub.add_parser("recalibrate", help="refit from feedback and report drift")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
    p.set_defaults(fn=cmd_recalibrate)

    p = sub.add_parser("fit", help="fit conformal thresholds")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
    p.add_argument("--report", action="store_true")
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("data", help="generate the seeded synthetic corpus")
    p.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
    p.add_argument("--scale", type=float, default=1.0)
    p.set_defaults(fn=cmd_data)

    p = sub.add_parser("eval", help="run the three-arm evaluation")
    p.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--dump-items", action="store_true")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("demo", help="offline A/B demo")
    p.add_argument("--scenario", default="compensation")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("serve", help="run the gateway and dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("info", help="print the active configuration")
    p.set_defaults(fn=cmd_info)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
