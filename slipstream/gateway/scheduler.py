"""Hold-back streaming scheduler — the differentiator (§7.1).

The mechanism in one paragraph: the provider generates tokens; we cut them into
chunks at sentence or clause boundaries; chunk N is **held** while its detectors
run, and those detectors run *concurrently with the provider generating chunk
N+1*. If the verdict is clean by the time chunk N+1 is cut, chunk N is released
and the verification cost me nothing in wall-clock. If it is dirty, chunk N is
never emitted and the action matrix fires. That concurrency is the entire point —
it is why an adaptive policy can afford deep verification on the traffic that
needs it without paying for it on the traffic that does not.

Three scheduling regimes, one code path:

* ``holdback_tokens == 0``  — tokens are emitted the instant they arrive and
  detectors scan behind them. Fast, and structurally unable to prevent a leak.
  This is the ``fixed_async`` baseline.
* ``sync == True``          — chunk N's verdict is awaited before generation of
  chunk N+1 continues. Safe, and pays full guard latency serially. This is the
  ``fixed_sync`` baseline.
* otherwise                 — hold-back with overlap. This is Slipstream.

**Backpressure:** if guards lose the race to generation for ``k`` consecutive
chunks the scheduler degrades to floor-only enforcement, marks the record
``degraded_mode=True`` and keeps going. It never stalls the stream silently.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

from ..actions.masking import mask_entities
from ..actions.matrix import ActionDecision, decide
from ..contracts import (
    Action,
    Band,
    ChunkTiming,
    GuardContext,
    RiskCategory,
    StreamPolicy,
    Verdict,
    max_severity,
)
from ..guards import consistency as consistency_guard
from ..guards.registry import DetectorRegistry
from ..policy.loader import PolicyPack
from ..textutil import approx_tokens, find_boundary
from .providers.base import GenerationRequest, Provider

# Chars per token, used to turn a holdback expressed in tokens into a buffer
# depth in characters. Deliberately a constant rather than a tokeniser call: the
# scheduler must not spend milliseconds deciding where to spend milliseconds.
CHARS_PER_TOKEN = 4

# INV-1 says the floor runs **inline** in every band, including the lowest. So
# ``holdback_tokens == 0`` cannot mean "emit with nothing checked" — it means "as
# shallow as the unconditional floor allows". Every band therefore holds at least
# this many tokens so the floor detectors can clear them before release.
#
# The one exception is the ``fixed_async`` arm, which is constructed with
# ``enforce_floor_inline=False`` precisely because being structurally unable to
# prevent a leak is the property that arm exists to measure.
FLOOR_HOLDBACK_TOKENS = 16

EventType = Literal["chunk", "annotation", "blocked", "escalated", "done"]


@dataclass
class StreamEvent:
    type: EventType
    text: str = ""
    index: int = 0
    masked_types: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class _Pending:
    index: int
    text: str
    task: asyncio.Task
    cut_at: float
    prev_cut_at: float
    degraded: bool


@dataclass
class StreamResult:
    """Everything the ledger and the eval need. Populated as the stream runs."""

    text: str = ""
    released_text: str = ""
    verdicts: list[Verdict] = field(default_factory=list)
    action: Action = Action.ALLOW
    reasons: list[str] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)
    entities_masked: list[str] = field(default_factory=list)
    chunk_timings: list[ChunkTiming] = field(default_factory=list)
    # Predicted label SET, not an argmax. Multi-label metrics read this (§9).
    triggered: set[RiskCategory] = field(default_factory=set)
    abstained_categories: set[RiskCategory] = field(default_factory=set)

    degraded_mode: bool = False
    abstained: bool = False
    regenerated: bool = False
    escalated: bool = False
    blocked: bool = False
    leaked_chunks: int = 0
    # Entity types that reached the user unmasked. Kept separate from
    # ``entities_masked`` so the two can never be confused in a metric.
    leaked_entity_types: list[str] = field(default_factory=list)
    masked_before_emission: bool = False

    ttft_ms: float = 0.0
    first_token_ms: float = 0.0
    total_latency_ms: float = 0.0
    generation_ms: float = 0.0
    guard_overhead_ms: float = 0.0
    guard_wall_ms: float = 0.0
    prompt_guard_ms: float = 0.0
    tokens_out: int = 0
    chunks: int = 0

    @property
    def added_latency_ms(self) -> float:
        """Wall-clock the gateway added on top of raw generation."""
        return max(0.0, self.total_latency_ms - self.generation_ms)

    @property
    def overlap_counts(self) -> tuple[int, int]:
        """(chunks whose guard won the race, chunks that HAD a race to win).

        The final chunk has no successor to overlap with, so it is excluded. A
        request with only one chunk contributes (0, 0) and must not be counted as
        a win — that would inflate the headline metric with requests that never
        exercised the mechanism.
        """
        overlapped = [
            t for t in self.chunk_timings if t.raced and t.index < self.chunks - 1
        ]
        return sum(1 for t in overlapped if t.guard_finished_first), len(overlapped)

    @property
    def pct_guard_finished_first(self) -> float:
        won, total = self.overlap_counts
        return won / total if total else 0.0


class Scheduler:
    def __init__(
        self,
        provider: Provider,
        registry: DetectorRegistry,
        pack: PolicyPack,
        *,
        backpressure_k: int = 3,
        scan_chunk_tokens: int = 48,
        enforce_floor_inline: bool = True,
        enforce_budget: bool = True,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.pack = pack
        # 0 disables degradation entirely. Backpressure is a *load-shedding*
        # response, so whether it fires depends on wall-clock and therefore on
        # machine load — which makes the decisions it changes non-reproducible.
        # Turning it off is how a run becomes bit-for-bit repeatable.
        self.backpressure_k = backpressure_k
        self.scan_chunk_tokens = scan_chunk_tokens
        self.enforce_floor_inline = enforce_floor_inline
        # Guard budgets are the other load-dependent input to a decision: under
        # load a cheap detector can overrun its budget, abstain, and change the
        # action. Real enforcement, and on by default — but it has to be
        # switchable for a bit-for-bit reproducible run.
        self.enforce_budget = enforce_budget

    # -- prompt side -----------------------------------------------------
    async def check_prompt(self, prompt: str, ctx: GuardContext) -> tuple[list[Verdict], ActionDecision]:
        """Floor detectors on the way in. Runs in every band (INV-1)."""
        detectors = self.registry.select([], side="prompt")
        verdicts = await self.registry.run(detectors, prompt, ctx)
        decision = decide(
            verdicts,
            self.pack,
            band=ctx.band,
            verifiable=ctx.verifiable,
            masking_enabled=True,
            is_prompt_side=True,
        )
        return verdicts, decision

    # -- output side -----------------------------------------------------
    async def stream(
        self,
        req: GenerationRequest,
        policy: StreamPolicy,
        ctx: GuardContext,
        result: StreamResult,
    ) -> AsyncIterator[StreamEvent]:
        """Drive the provider and release verified chunks."""
        sampler_token = consistency_guard.current_sampler.set(
            lambda n: self.provider.sample(req, n)
        )
        try:
            async for ev in self._stream(req, policy, ctx, result):
                yield ev
        finally:
            consistency_guard.current_sampler.reset(sampler_token)

    async def _stream(
        self,
        req: GenerationRequest,
        policy: StreamPolicy,
        ctx: GuardContext,
        result: StreamResult,
    ) -> AsyncIterator[StreamEvent]:
        t_start = time.perf_counter()
        if policy.holdback_tokens == 0 and self.enforce_floor_inline:
            # "stream_first, subject to the unconditional floor" (INV-1).
            emit_immediately = False
            holdback = FLOOR_HOLDBACK_TOKENS
        else:
            emit_immediately = policy.holdback_tokens == 0
            holdback = policy.holdback_tokens
        min_chars = max(
            CHARS_PER_TOKEN,
            (holdback or self.scan_chunk_tokens) * CHARS_PER_TOKEN,
        )

        buf = ""
        full = ""
        pending: _Pending | None = None
        background: list[tuple[int, str, asyncio.Task, float]] = []
        chunk_index = 0
        emitted_index = 0
        consecutive_losses = 0
        degraded = False
        last_cut_at = t_start
        first_token_at: float | None = None
        candidate_actions: list[Action] = [Action.ALLOW]

        async def release(p: _Pending) -> AsyncIterator[StreamEvent]:
            """Await chunk p's verdict, apply the action, emit if permitted."""
            nonlocal consecutive_losses, degraded, emitted_index
            finished_first = p.task.done()
            guard_t0 = time.perf_counter()
            verdicts = await p.task
            wait_ms = (time.perf_counter() - guard_t0) * 1000.0
            guard_ms = max(v.latency_ms for v in verdicts) if verdicts else 0.0
            gen_ms = (p.cut_at - p.prev_cut_at) * 1000.0

            result.verdicts.extend(verdicts)
            result.guard_wall_ms += guard_ms
            result.guard_overhead_ms += wait_ms

            # A synchronous policy has no concurrent window, so it cannot "lose"
            # a race it never ran. Counting those as losses would trip
            # backpressure on every sync request after three chunks.
            raced = not policy.sync
            if raced and not finished_first:
                consecutive_losses += 1
                if (
                    self.backpressure_k > 0
                    and consecutive_losses >= self.backpressure_k
                    and not degraded
                ):
                    degraded = True
                    result.degraded_mode = True
                    result.reasons.append(
                        f"backpressure: guards lost {consecutive_losses} consecutive "
                        f"chunks -> floor-only enforcement"
                    )
            elif raced:
                consecutive_losses = 0

            d = decide(
                verdicts,
                self.pack,
                band=policy.band,
                verifiable=ctx.verifiable,
                masking_enabled=policy.masking_enabled,
            )
            result.reasons.extend(d.reasons)
            result.annotations.extend(a for a in d.annotations if a not in result.annotations)
            result.triggered |= d.triggered
            result.abstained_categories |= d.abstained
            if d.abstained:
                result.abstained = True
            candidate_actions.append(d.action)

            text_out = p.text
            released = True
            # Masking is a TRANSFORMATION, not a rung on the severity ladder. If
            # PII is present and masking is on, the entities come out — whatever
            # else the matrix decided. Gating this on `action is MASK` meant a
            # chunk that also tripped groundedness (REGENERATE, higher severity)
            # was released with the PII intact. INV-1 does not bend to rule
            # ordering.
            if d.entities_to_mask and policy.masking_enabled:
                text_out, types = mask_entities(p.text, d.entities_to_mask)
                result.entities_masked = sorted(set(result.entities_masked) | set(types))
                result.masked_before_emission = True
            if d.action is Action.BLOCK:
                released = False
                result.blocked = True
            elif d.action is Action.ESCALATE:
                released = False
                result.escalated = True

            result.chunk_timings.append(
                ChunkTiming(
                    index=p.index,
                    tokens=approx_tokens(p.text),
                    gen_ms=gen_ms,
                    guard_ms=guard_ms,
                    guard_finished_first=finished_first,
                    raced=raced,
                    released=released,
                    leaked=False,
                    degraded=p.degraded,
                )
            )

            if released:
                result.released_text += text_out
                yield StreamEvent(
                    type="chunk",
                    text=text_out,
                    index=emitted_index,
                    masked_types=sorted({e.entity_type for e in d.entities_to_mask}),
                    meta={"band": policy.band.value, "degraded": p.degraded},
                )
                emitted_index += 1
                if result.ttft_ms == 0.0:
                    result.ttft_ms = (time.perf_counter() - t_start) * 1000.0
            else:
                yield StreamEvent(
                    type="blocked" if d.action is Action.BLOCK else "escalated",
                    text=d.fallback_text or "",
                    index=p.index,
                    meta={"reasons": d.reasons},
                )

        def launch(text: str, index: int, cut_at: float, prev_cut_at: float) -> _Pending:
            detectors = self.registry.select(
                policy.detectors, side="output", degraded=degraded
            )
            chunk_ctx = ctx.model_copy(update={"budget_ms": policy.guard_budget_ms})
            task = asyncio.create_task(
                self.registry.run(
                    detectors,
                    text,
                    chunk_ctx,
                    budget_ms=policy.guard_budget_ms if self.enforce_budget else None,
                )
            )
            return _Pending(index, text, task, cut_at, prev_cut_at, degraded)

        # ---- token loop -------------------------------------------------
        async for tok in self.provider.stream(req):
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
                result.first_token_ms = (now - t_start) * 1000.0
            buf += tok.text
            full += tok.text

            if emit_immediately:
                # fixed_async: the token is on the user's screen before anything
                # has looked at it. Scanning happens behind, on a chunk cadence.
                result.released_text += tok.text
                if result.ttft_ms == 0.0:
                    result.ttft_ms = (now - t_start) * 1000.0
                yield StreamEvent(type="chunk", text=tok.text, index=emitted_index)
                emitted_index += 1

            cut = find_boundary(buf, min_chars)
            if not cut:
                continue

            chunk_text, buf = buf[:cut], buf[cut:]

            if emit_immediately:
                p = launch(chunk_text, chunk_index, now, last_cut_at)
                background.append((chunk_index, chunk_text, p.task, now))
                chunk_index += 1
                last_cut_at = now
                continue

            # Resolve the previous chunk first: its guard has been running for
            # the whole time the provider spent generating THIS chunk. That gap
            # is the latency the design is buying back.
            if pending is not None:
                async for ev in release(pending):
                    yield ev
                pending = None
                if result.blocked or result.escalated:
                    break

            pending = launch(chunk_text, chunk_index, now, last_cut_at)
            chunk_index += 1
            last_cut_at = now

            if policy.sync:
                # fixed_sync: block generation until this chunk clears.
                async for ev in release(pending):
                    yield ev
                pending = None
                if result.blocked or result.escalated:
                    break

        # Raw generation cost: everything up to the provider's last token. What
        # the gateway adds on top of this is the number the eval reports.
        result.generation_ms = (time.perf_counter() - t_start) * 1000.0

        # ---- drain ------------------------------------------------------
        if not (result.blocked or result.escalated):
            if pending is not None:
                async for ev in release(pending):
                    yield ev
                pending = None

            if buf.strip() and not (result.blocked or result.escalated):
                now = time.perf_counter()
                tail = launch(buf, chunk_index, now, last_cut_at)
                chunk_index += 1
                if emit_immediately:
                    # Those tokens were already on the screen the moment they
                    # arrived. Scan the tail, never re-emit it.
                    background.append((tail.index, buf, tail.task, now))
                else:
                    async for ev in release(tail):
                        yield ev
                buf = ""

        # Background scans for the async arm. Everything here is a post-hoc
        # finding: if it is dirty, it already reached the user.
        if background:
            await self._drain_background(background, policy, ctx, result, candidate_actions)

        result.text = full
        result.tokens_out = approx_tokens(full)
        result.chunks = chunk_index
        result.action = max_severity(candidate_actions)
        result.total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        yield StreamEvent(
            type="done",
            meta={
                "action": result.action.value,
                "degraded": result.degraded_mode,
                "leaked_chunks": result.leaked_chunks,
            },
        )

    async def _drain_background(
        self,
        background: list[tuple[int, str, asyncio.Task, float]],
        policy: StreamPolicy,
        ctx: GuardContext,
        result: StreamResult,
        candidate_actions: list[Action],
    ) -> None:
        prev_cut = None
        for index, text, task, cut_at in background:
            verdicts = await task
            result.verdicts.extend(verdicts)
            guard_ms = max((v.latency_ms for v in verdicts), default=0.0)
            d = decide(
                verdicts,
                self.pack,
                band=policy.band,
                verifiable=ctx.verifiable,
                masking_enabled=policy.masking_enabled,
            )
            result.reasons.extend(f"post_hoc:{r}" for r in d.reasons)
            result.annotations.extend(
                a for a in d.annotations if a not in result.annotations
            )
            result.triggered |= d.triggered
            result.abstained_categories |= d.abstained
            if d.abstained:
                result.abstained = True
            candidate_actions.append(d.action)

            # A leak is "content that would have been withheld or transformed had
            # it been held back". Keying this on d.action would miss the case
            # where a higher-severity action outranked MASK — the PII still
            # reached the user. `entities_masked` stays empty here on purpose:
            # nothing was masked, and recording types would imply otherwise.
            leaked = bool(d.entities_to_mask) or d.action in (
                Action.BLOCK,
                Action.ESCALATE,
            )
            if leaked:
                result.leaked_chunks += 1
                result.leaked_entity_types = sorted(
                    set(result.leaked_entity_types)
                    | {e.entity_type for e in d.entities_to_mask}
                )
            gen_ms = ((cut_at - prev_cut) * 1000.0) if prev_cut else 0.0
            prev_cut = cut_at
            result.guard_wall_ms += guard_ms
            result.chunk_timings.append(
                ChunkTiming(
                    index=index,
                    tokens=approx_tokens(text),
                    gen_ms=gen_ms,
                    guard_ms=guard_ms,
                    guard_finished_first=task.done(),
                    released=True,
                    leaked=leaked,
                    degraded=False,
                )
            )


def policy_for_arm(
    arm: str, policy: StreamPolicy, pack: PolicyPack, scan_chunk_tokens: int = 48
) -> StreamPolicy:
    """Rewrite a policy for a baseline arm.

    INV-3: only the *scheduling* fields change. Detector list, masking and
    escalation arming are carried over untouched so all arms enforce the same
    thing with the same models at the same thresholds.
    """
    if arm == "adaptive":
        return policy
    if arm == "fixed_sync":
        depth = pack.band_config(Band.HIGH).holdback_tokens or 256
        return policy.model_copy(
            update={
                "holdback_tokens": depth,
                "sync": True,
                "detectors": pack.detectors_for(Band.HIGH),
                "guard_budget_ms": pack.band_config(Band.HIGH).guard_budget_ms,
            }
        )
    if arm == "fixed_async":
        return policy.model_copy(
            update={
                "holdback_tokens": 0,
                "sync": False,
                "detectors": pack.detectors_for(Band.HIGH),
                "guard_budget_ms": pack.band_config(Band.HIGH).guard_budget_ms,
            }
        )
    if arm == "nemo_default":
        # NeMo Guardrails streaming defaults: chunk_size 200, context_size 50.
        return policy.model_copy(
            update={
                "holdback_tokens": 200,
                "sync": False,
                "detectors": pack.detectors_for(Band.HIGH),
                "guard_budget_ms": pack.band_config(Band.HIGH).guard_budget_ms,
            }
        )
    raise ValueError(f"unknown arm {arm!r}")
