"""Headless A/B demo (§11), for the terminal and for the video voiceover.

    python -m slipstream.demo.run
    python -m slipstream.demo.run --scenario injection

Runs the same request through `fixed_async` and `adaptive` with the same
detectors and prints, side by side, what actually reached the user plus the live
TTFT / added-latency counters and the decision record's place in the hash chain.
Offline, no keys.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from .. import settings
from ..guards.registry import default_registry
from ..ledger.store import Ledger, verify_chain
from ..policy.loader import get_loader
from ..gateway.pipeline import SlipstreamGateway, SlipstreamRequest
from ..gateway.providers.replay import ReplayProvider
from .scenario import DEMO_FIXTURES, SCENARIOS, get_scenario

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def _wrap(text: str, width: int = 78, indent: str = "    ") -> str:
    import textwrap

    return "\n".join(
        textwrap.fill(p, width=width, initial_indent=indent, subsequent_indent=indent)
        for p in text.split("\n")
        if p.strip()
    ) or f"{indent}(nothing was released)"


async def run_arm(gateway: SlipstreamGateway, scenario: dict, arm: str) -> dict:
    req = SlipstreamRequest(
        prompt=scenario["prompt"],
        use_case=scenario["use_case"],
        geography=scenario.get("geography", "IN"),
        retrieval_context=list(scenario.get("retrieval_context", [])),
        arm=arm,
        fixture_id=scenario["fixture_id"],
        use_cache=False,
    )
    t0 = time.perf_counter()
    emitted: list[str] = []
    events: list[str] = []
    first_visible_ms = None
    # What was on the screen before a regeneration replaced it. Erasing a
    # response after the fact does not un-display it, and the demo's whole point
    # is what the user actually saw.
    replaced_text = ""
    replaced_at_ms = 0.0
    async for ev in gateway.run(req):
        if ev.type == "chunk":
            if first_visible_ms is None:
                first_visible_ms = (time.perf_counter() - t0) * 1000.0
            emitted.append(ev.text)
        elif ev.type in ("blocked", "escalated"):
            events.append(f"{ev.type}: {ev.text}")
        elif ev.type == "annotation":
            events.append(f"annotation: {ev.text}")
            if ev.meta.get("restart"):
                replaced_text = "".join(emitted)
                replaced_at_ms = (time.perf_counter() - t0) * 1000.0
                emitted.clear()
    out = gateway.last_outcome
    return {
        "arm": arm,
        "text": "".join(emitted),
        "replaced_text": replaced_text,
        "replaced_at_ms": replaced_at_ms,
        "events": events,
        "outcome": out,
        "first_visible_ms": first_visible_ms or 0.0,
    }


def print_pane(r: dict, colour: str) -> None:
    o = r["outcome"]
    rec, res = o.record, o.result
    leaked = res.leaked_chunks > 0
    header = f" {r['arm'].upper()} "
    print(f"{colour}{BOLD}{header:=^80}{RESET}")
    print(
        f"  band={rec.policy.band.value}  holdback={rec.policy.holdback_tokens} tok  "
        f"sync={rec.policy.sync}  detectors={','.join(rec.policy.detectors)}"
    )
    print(f"  {DIM}band reason: {o.band_reason}{RESET}")
    print(
        f"  TTFT {r['first_visible_ms']:.0f} ms   total {rec.total_latency_ms:.0f} ms   "
        f"added {res.added_latency_ms:.0f} ms   guard overhead {rec.guard_overhead_ms:.0f} ms"
    )
    print(f"  action={BOLD}{rec.action.value}{RESET}   masked={rec.entities_masked or '[]'}")
    if leaked:
        print(
            f"  {RED}{BOLD}LEAKED: {res.leaked_chunks} chunk(s) reached the user "
            f"before their verdict existed "
            f"({', '.join(res.leaked_entity_types) or 'unmaskable content'}){RESET}"
        )
    elif rec.entities_masked:
        print(f"  {GREEN}{BOLD}MASKED BEFORE EMISSION: nothing sensitive was displayed{RESET}")
    print()
    if r["replaced_text"]:
        print(
            f"{DIM}  what the user saw first (replaced after "
            f"{r['replaced_at_ms']:.0f} ms):{RESET}"
        )
        print(f"{RED}{_wrap(r['replaced_text'])}{RESET}")
        print()
        print(f"{DIM}  what replaced it:{RESET}")
    else:
        print(f"{DIM}  what the user saw:{RESET}")
    print(_wrap(r["text"]))
    for e in r["events"]:
        print(f"  {YELLOW}{e}{RESET}")
    print()
    print(f"{DIM}  decision record {rec.record_id[:12]}…{RESET}")
    print(f"{DIM}    prev_hash   {rec.prev_hash[:32]}…{RESET}")
    print(f"{DIM}    record_hash {rec.record_hash[:32]}…{RESET}")
    print()


async def main_async(args) -> int:
    settings.ensure_dirs()
    scenario = get_scenario(args.scenario)

    provider = ReplayProvider(DEMO_FIXTURES)
    ledger = Ledger(settings.DATA_DIR / "demo_ledger.sqlite3")
    gateway = SlipstreamGateway(
        provider,
        registry=default_registry(),
        loader=get_loader(),
        ledger=ledger,
        emit_telemetry=True,
    )

    print()
    print(f"{BOLD}SLIPSTREAM — A/B demo{RESET}")
    print(f"  scenario : {scenario['title']}")
    print(f"  prompt   : {scenario['prompt']}")
    print(f"  {DIM}{scenario['note']}{RESET}")
    print(
        f"  {DIM}both panes: identical detectors "
        f"({default_registry().fingerprint()}), identical thresholds, "
        f"identical provider. Only the scheduling policy differs (INV-3).{RESET}"
    )
    print()

    left = await run_arm(gateway, scenario, "fixed_async")
    right = await run_arm(gateway, scenario, "adaptive")

    print_pane(left, RED)
    print_pane(right, GREEN)

    brk = verify_chain(ledger)
    print(f"{BOLD}{'  ledger  ':=^80}{RESET}")
    print(f"  {len(ledger)} record(s)")
    if brk is None:
        print(f"  {GREEN}hash chain intact{RESET}")
    else:
        print(f"  {RED}chain break at seq {brk.seq}: {brk.kind}{RESET}")
    print(f"  {DIM}spans -> {settings.OTEL_FILE}{RESET}")
    print()

    delta = left["first_visible_ms"] - right["first_visible_ms"]
    print(f"{BOLD}{'  the point  ':=^80}{RESET}")
    if right["outcome"].record.entities_masked and left["outcome"].result.leaked_chunks:
        print(
            f"  Same detectors, same thresholds, same model. `fixed_async` put "
            f"{', '.join(left['outcome'].result.leaked_entity_types)} on the screen "
            f"and flagged it afterwards.\n  `adaptive` scored the prompt "
            f"{right['outcome'].record.policy.band.value.upper()}, deepened the "
            f"buffer to {right['outcome'].record.policy.holdback_tokens} tokens, and "
            f"masked every one of them before the first chunk was released — for "
            f"{abs(delta):.0f} ms of extra time to first token."
        )
    else:
        print(
            f"  Both arms reached action "
            f"`{right['outcome'].record.action.value}`. TTFT differed by "
            f"{abs(delta):.0f} ms."
        )
    print()
    ledger.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Slipstream offline A/B demo")
    ap.add_argument("--scenario", default="compensation", choices=sorted(SCENARIOS))
    args = ap.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
