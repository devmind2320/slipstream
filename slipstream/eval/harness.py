"""Three-arm evaluation harness (§10).

    python -m slipstream.eval.harness --seed 1729

The arms differ **only in scheduling policy** (INV-3). They are handed the same
``DetectorRegistry`` *instance*, the same fitted thresholds, the same provider and
the same items; the harness asserts the registry fingerprint is identical across
arms and records it in results.json so the claim is checkable, not asserted.

Run order:

1. **Score** every item with the full detector set. Detector scores do not depend
   on scheduling, so this pass is arm-independent and is what feeds calibration.
2. **Fit** split-conformal thresholds on the *calibration* split only, and apply
   them to every pack. The test split never touches the fit.
3. **Run** each arm end-to-end over the *test* split.
4. **Probe** the cache separately, with exact repeats and entity-perturbed
   near-misses, because the main corpus has no repeated prompts and measuring
   cache behaviour on it would produce a meaningless zero.
5. **Report**.

On determinism (§13): decisions are deterministic and are digested into
``deterministic_digest``, which two seeded runs must match byte for byte.
Wall-clock latency is not deterministic — it depends on the machine — so it lives
in its own block and is excluded from that digest. Claiming otherwise would be
the kind of unreproducible number INV-4 exists to prevent.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .. import settings
from ..calibration.conformal import (
    CalibrationItem,
    append_calibration_items,
    fit_all,
    save_thresholds,
)
from ..contracts import Action, GuardContext, RiskCategory
from ..gateway.pipeline import SlipstreamGateway, SlipstreamRequest
from ..gateway.providers.base import GenerationRequest
from ..gateway.providers.replay import ReplayProvider
from ..guards import consistency as consistency_guard
from ..guards.registry import build_registry
from ..policy.loader import PolicyLoader
from ..risk.conversation import ConversationStore
from ..risk.features import ToolScope
from ..routing.cache import SemanticCache, entity_signature
from .datasets import generate as gen
from .metrics import ItemResult, compute_arm_metrics, sanitize
from .report import render_report

ARMS = ("fixed_sync", "fixed_async", "adaptive")


# ---------------------------------------------------------------------------
# 1. Scoring pass
# ---------------------------------------------------------------------------
async def score_item(registry, provider, loader, item: gen.Item) -> dict[str, float]:
    """Full detector set on one item. Arm-independent by construction."""
    pack = loader.load(item.use_case)
    ctx = GuardContext(
        prompt=item.prompt,
        retrieval_context=item.retrieval_context,
        use_case=item.use_case,
        geography=item.geography,
        pii_entities=pack.pii_entities(item.geography),
        protected_attributes=pack.bias.protected_attributes,
        decision_context=pack.bias.decision_context,
        capabilities=provider.capabilities(),
    )
    everything = [registry.get(n) for n in registry.names]
    prompt_side = [d for d in everything if getattr(d, "side", "both") in ("both", "prompt")]
    output_side = [d for d in everything if getattr(d, "side", "both") in ("both", "output")]

    token = consistency_guard.current_sampler.set(
        lambda n: provider.sample(
            GenerationRequest(prompt=item.prompt, fixture_id=item.item_id), n
        )
    )
    try:
        pv = await registry.run(
            prompt_side, item.prompt, ctx.model_copy(update={"is_prompt_side": True})
        )
        ov = await registry.run(output_side, item.response, ctx)
    finally:
        consistency_guard.current_sampler.reset(token)

    scores: dict[str, float] = {c.value: 0.0 for c in RiskCategory}
    # Injection is judged on the way in; everything else on the way out.
    for v in pv:
        if v.abstained or v.category is not RiskCategory.INJECTION:
            continue
        scores[v.category.value] = max(scores[v.category.value], v.score)
    for v in ov:
        if v.abstained or v.category is RiskCategory.INJECTION:
            continue
        scores[v.category.value] = max(scores[v.category.value], v.score)
    return scores


async def scoring_pass(
    registry, provider, loader, items: list[gen.Item], concurrency: int
) -> dict[str, dict[str, float]]:
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, dict[str, float]] = {}

    async def one(it: gen.Item) -> None:
        async with sem:
            out[it.item_id] = await score_item(registry, provider, loader, it)

    await asyncio.gather(*(one(i) for i in items))
    return out


def to_calibration_items(
    items: list[gen.Item], scores: dict[str, dict[str, float]]
) -> list[CalibrationItem]:
    """One labelled example per (item, category). Labels are set-membership."""
    out: list[CalibrationItem] = []
    for it in items:
        if it.split != "calibration":
            continue
        s = scores.get(it.item_id, {})
        for cat in (c.value for c in RiskCategory):
            out.append(
                CalibrationItem(
                    use_case=it.use_case,
                    category=cat,
                    score=round(s.get(cat, 0.0), 4),
                    label=1 if cat in it.labels else 0,
                    source="eval",
                    item_id=it.item_id,
                )
            )
    return out


# ---------------------------------------------------------------------------
# 3. Arm run
# ---------------------------------------------------------------------------
async def run_arm(
    arm: str,
    items: list[gen.Item],
    *,
    registry,
    provider,
    loader,
    scores: dict[str, dict[str, float]],
    concurrency: int,
    cache: SemanticCache | None = None,
    backpressure_k: int = 3,
    enforce_budget: bool = True,
) -> list[ItemResult]:
    gateway = SlipstreamGateway(
        provider,
        registry=registry,
        loader=loader,
        ledger=None,
        cache=cache,
        conversations=ConversationStore(),
        emit_telemetry=False,
        backpressure_k=backpressure_k,
        enforce_budget=enforce_budget,
    )
    sem = asyncio.Semaphore(concurrency)
    results: list[ItemResult] = []

    singles = [i for i in items if not i.conversation_id]
    sessions: dict[str, list[gen.Item]] = {}
    for i in items:
        if i.conversation_id:
            sessions.setdefault(i.conversation_id, []).append(i)

    def to_result(it: gen.Item, outcome) -> ItemResult:
        rec, res = outcome.record, outcome.result
        won, total = res.overlap_counts
        return ItemResult(
            item_id=it.item_id,
            slice=it.slice,
            use_case=it.use_case,
            arm=arm,
            band=rec.policy.band.value,
            labels=sorted(it.labels),
            predicted=sorted(c.value for c in res.triggered),
            action=rec.action.value,
            signal_mode=rec.signal_mode.value,
            ttft_ms=rec.ttft_ms,
            first_token_ms=res.first_token_ms,
            total_latency_ms=rec.total_latency_ms,
            generation_ms=res.generation_ms,
            guard_overhead_ms=rec.guard_overhead_ms,
            masked_before_emission=res.masked_before_emission,
            entities_masked=list(rec.entities_masked),
            leaked_chunks=res.leaked_chunks,
            abstained=res.abstained,
            expects_abstention=it.expects_abstention,
            degraded=res.degraded_mode,
            escalated=res.escalated,
            regenerated=res.regenerated,
            budget_timeouts=sum(
                1 for v in res.verdicts if v.evidence == "budget_exceeded"
            ),
            guard_first_won=won,
            guard_first_total=total,
            tokens_in=rec.tokens_in,
            tokens_out=rec.tokens_out,
            model_used=rec.model_used,
            routing_tier=it.routing_tier,
            baseline_cost=outcome.baseline_cost,
            actual_cost=outcome.actual_cost,
            cache_outcome=outcome.cache_outcome,
            category_scores=scores.get(it.item_id, {}),
        )

    async def run_single(it: gen.Item) -> None:
        async with sem:
            outcome = await gateway.collect(_request(it, arm))
            results.append(to_result(it, outcome))

    async def run_session(turns: list[gen.Item]) -> None:
        # Turn order matters: the ratchet is the thing under test.
        async with sem:
            for it in sorted(turns, key=lambda x: x.turn_index):
                outcome = await gateway.collect(_request(it, arm))
                results.append(to_result(it, outcome))

    await asyncio.gather(
        *(run_single(i) for i in singles),
        *(run_session(t) for t in sessions.values()),
    )
    results.sort(key=lambda r: r.item_id)
    return results


def _request(it: gen.Item, arm: str) -> SlipstreamRequest:
    return SlipstreamRequest(
        prompt=it.prompt,
        use_case=it.use_case,
        geography=it.geography,
        retrieval_context=list(it.retrieval_context),
        conversation_id=it.conversation_id,
        tool_scope=ToolScope(it.tool_scope),
        arm=arm,
        fixture_id=it.item_id,
        use_cache=False,  # measured separately; see cache_probe
    )


# ---------------------------------------------------------------------------
# 4. Cache probe
# ---------------------------------------------------------------------------
def _perturb(prompt: str) -> str | None:
    """Change exactly one identifier and nothing else.

    Returns None when the prompt carries no identifier to change. Those prompts
    are NOT near-misses — two requests that differ only in wording and agree on
    every name, figure and identifier *should* share a cache entry — and
    counting them as gate failures would measure the wrong thing entirely.
    """
    import re

    from ..eval.datasets.generate import FIRST

    def bump(m):
        return str(int(m.group(0)) + 7)

    out, n = re.subn(r"\b\d{3,}\b", bump, prompt, count=1)
    if n:
        return out
    for name in FIRST:
        if name in prompt:
            replacement = FIRST[(FIRST.index(name) + 5) % len(FIRST)]
            return prompt.replace(name, replacement, 1)
    return None


async def cache_probe(
    items: list[gen.Item], *, registry, provider, loader, n: int = 60
) -> dict:
    """Exact repeats should hit; entity-perturbed near-misses must NOT.

    Run on its own because the main corpus contains no repeated prompts, so
    measuring the cache there would report a meaningless zero.
    """
    cache = SemanticCache(similarity_threshold=0.90)
    gateway = SlipstreamGateway(
        provider, registry=registry, loader=loader, cache=cache, emit_telemetry=False
    )
    # Sample prompts that actually CARRY an identifier. A probe drawn from
    # prompts with no name, figure or account number in them cannot construct a
    # near-miss at all, and would report a vacuous zero.
    candidates = [
        i
        for i in items
        if not i.conversation_id and _perturb(i.prompt) is not None
    ]
    by_slice: dict[str, list] = {}
    for i in candidates:
        by_slice.setdefault(i.slice, []).append(i)
    sample: list = []
    while len(sample) < n and any(by_slice.values()):
        for slice_items in by_slice.values():  # round-robin, so no slice dominates
            if slice_items and len(sample) < n:
                sample.append(slice_items.pop(0))

    for it in sample:  # cold pass: populate
        await gateway.collect(_cached_request(it, it.prompt))
    cold = cache.stats.as_dict()

    cache.stats = type(cache.stats)()
    for it in sample:  # exact repeats
        await gateway.collect(_cached_request(it, it.prompt))
    repeat = cache.stats.as_dict()

    cache.stats = type(cache.stats)()
    near_miss_served = 0
    n_true_near_misses = 0
    skipped_no_identifier = 0
    for it in sample:  # entity-perturbed near misses
        perturbed = _perturb(it.prompt)
        if perturbed is None or perturbed == it.prompt:
            skipped_no_identifier += 1
            continue
        # Only count it if the perturbation genuinely moved an identifier;
        # otherwise a hit is correct behaviour, not a gate failure.
        if entity_signature(perturbed) == entity_signature(it.prompt):
            skipped_no_identifier += 1
            continue
        n_true_near_misses += 1
        outcome = await gateway.collect(_cached_request(it, perturbed))
        if outcome.record.cache_hit:
            near_miss_served += 1
    near = cache.stats.as_dict()

    return {
        "n_probed": len(sample),
        "cold_pass": cold,
        "exact_repeat_pass": repeat,
        "near_miss_pass": near,
        "n_true_near_misses": n_true_near_misses,
        "skipped_no_identifier_to_change": skipped_no_identifier,
        "near_miss_wrongly_served": near_miss_served,
        "gated_misses_on_near_miss": near["gated_misses"],
    }


def _cached_request(it: gen.Item, prompt: str) -> SlipstreamRequest:
    return SlipstreamRequest(
        prompt=prompt,
        use_case=it.use_case,
        geography=it.geography,
        retrieval_context=list(it.retrieval_context),
        arm="adaptive",
        fixture_id=it.item_id,
        use_cache=True,
    )


# ---------------------------------------------------------------------------
# NeMo arm
# ---------------------------------------------------------------------------
def nemo_status() -> dict:
    try:
        import nemoguardrails  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return {
            "run": False,
            "reason": (
                f"nemoguardrails is not importable in this environment "
                f"({type(exc).__name__}). The fourth arm was skipped rather than "
                f"faked, per spec §10."
            ),
        }
    return {
        "run": True,
        "reason": (
            "nemoguardrails imported successfully. NOTE: the nemo_default arm runs "
            "the Slipstream scheduler configured at NeMo's documented streaming "
            "defaults (chunk_size 200, context_size 50) so that INV-3 holds — same "
            "detectors, same thresholds, only the chunking policy differs. It is "
            "NOT NeMo's own rails executing, and must not be read as one."
        ),
    }


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------
def deterministic_digest(by_arm: dict[str, list[ItemResult]]) -> dict[str, str]:
    """Digest over the decisions that are actually reproducible.

    Items whose decision depended on **wall-clock** are excluded. There are
    exactly two such mechanisms, and both are real enforcement rather than
    artefacts:

    * **backpressure degradation** — guards that lose the race to generation for
      k consecutive chunks drop the policy to floor-only;
    * **guard budget timeouts** — a detector that overruns its budget abstains.

    Whether either fires depends on machine load, and both change which
    detectors contributed to the decision. They are reported
    (`scheduler.degraded_mode_rate`, `scheduler.budget_timeout_rate`) but not
    claimed to be reproducible — pretending otherwise would give a determinism
    check that passes by luck.

    `--deterministic` disables both and makes every decision in the run
    reproducible; that is what the determinism test uses.
    """
    out: dict[str, str] = {}
    for arm, results in sorted(by_arm.items()):
        rows = [
            "|".join(
                [
                    r.item_id,
                    r.band,
                    r.action,
                    ",".join(r.predicted),
                    str(int(r.abstained)),
                    str(int(r.masked_before_emission)),
                    str(int(r.leaked_chunks > 0)),
                    r.model_used,
                ]
            )
            for r in sorted(results, key=lambda x: x.item_id)
            if not r.load_dependent
        ]
        out[arm] = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:16]
        out[f"{arm}__n_items_digested"] = str(len(rows))
        out[f"{arm}__n_excluded_load_dependent"] = str(len(results) - len(rows))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run(args) -> dict:
    settings.ensure_dirs()
    t_start = time.perf_counter()

    items = gen.generate(seed=args.seed, scale=args.scale)
    manifest = gen.write(items, settings.DATA_DIR, args.seed, args.scale)
    if args.limit:
        items = items[: args.limit]

    provider = ReplayProvider(
        [i.to_fixture() for i in items],
        supports_logprobs=False,  # the honest enterprise default (§7.4)
        supports_multi_sample=True,
    )
    # INV-3: ONE registry instance, shared by every arm.
    registry = build_registry()
    loader = PolicyLoader()
    budgets = {
        uc: loader.load(uc).escalation.budget for uc in loader.available()
    }

    print(f"[1/5] scoring {len(items)} items with the full detector set ...")
    scores = await scoring_pass(registry, provider, loader, items, args.concurrency)

    print("[2/5] fitting split-conformal thresholds on the calibration split ...")
    cal_items = to_calibration_items(items, scores)
    if settings.CALIBRATION_SET_PATH.exists():
        settings.CALIBRATION_SET_PATH.unlink()
    append_calibration_items(cal_items)
    calibration = fit_all(cal_items, alpha=args.alpha, beta_by_use_case=budgets, seed=args.seed)
    save_thresholds(calibration)
    loader.apply_calibration(calibration["thresholds"])

    test_items = [i for i in items if i.split == "test"]
    arms = list(args.arms)
    nemo = nemo_status()
    if nemo["run"]:
        arms.append("nemo_default")

    by_arm: dict[str, list[ItemResult]] = {}
    fingerprints: dict[str, str] = {}
    for n, arm in enumerate(arms, start=3):
        print(f"[3/5] arm {arm}: {len(test_items)} items ...")
        fingerprints[arm] = registry.fingerprint()
        by_arm[arm] = await run_arm(
            arm,
            test_items,
            registry=registry,
            provider=provider,
            loader=loader,
            scores=scores,
            concurrency=args.concurrency,
            backpressure_k=0 if args.deterministic else args.backpressure_k,
            enforce_budget=not args.deterministic,
        )

    assert len(set(fingerprints.values())) <= 1, (
        "INV-3 violated: arms did not share an identical detector registry"
    )

    print("[4/5] cache probe ...")
    cache_stats = await cache_probe(
        test_items, registry=registry, provider=provider, loader=loader
    )

    print("[5/5] computing metrics ...")
    payload = {
        "run": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "scale": args.scale,
            "alpha": args.alpha,
            "arms": arms,
            "n_test_items": len(test_items),
            "dataset": manifest,
            "backends": settings.backend_summary(),
            "registry_fingerprint": registry.fingerprint(),
            "registry_detectors": registry.names,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "wall_clock_s": None,
            "backpressure_k": 0 if args.deterministic else args.backpressure_k,
            "deterministic_mode": bool(args.deterministic),
            "reproduce": (
                f"python -m slipstream.eval.harness --seed {args.seed} "
                f"--scale {args.scale} --alpha {args.alpha}"
                + (" --deterministic" if args.deterministic else "")
            ),
        },
        "calibration": calibration,
        "arms": {arm: compute_arm_metrics(rs, budgets) for arm, rs in by_arm.items()},
        "cache_probe": cache_stats,
        "nemo_arm": nemo,
        "deterministic_digest": deterministic_digest(by_arm),
    }
    payload["run"]["wall_clock_s"] = round(time.perf_counter() - t_start, 1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = sanitize(payload)
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    if args.dump_items:
        with (out_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
            for arm, rs in sorted(by_arm.items()):
                for r in rs:
                    fh.write(json.dumps(asdict(r), sort_keys=True) + "\n")
    (out_dir / "report.md").write_text(render_report(payload), encoding="utf-8")

    print(f"\nresults -> {out_dir / 'results.json'}")
    print(f"report  -> {out_dir / 'report.md'}")
    print(f"digest  -> {payload['deterministic_digest']}")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Slipstream three-arm evaluation")
    ap.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument(
        "--backpressure-k",
        type=int,
        default=3,
        help="consecutive lost races before degrading to floor-only; 0 disables "
        "degradation and makes every decision in the run reproducible",
    )
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="disable backpressure degradation AND guard-budget timeouts, the two "
        "wall-clock-dependent inputs to a decision, so the run is bit-for-bit "
        "reproducible. Costs the scheduler's ability to shed load.",
    )
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", type=Path, default=settings.RESULTS_DIR)
    ap.add_argument("--dump-items", action="store_true")
    args = ap.parse_args(argv)
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
