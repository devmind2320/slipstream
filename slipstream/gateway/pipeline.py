"""Request orchestration: estimate -> route -> schedule -> act -> record.

Not in the spec's file list. It exists because the FastAPI app, the eval harness
and the demo all need the *same* end-to-end path, and the alternative was three
copies of it — which would have made INV-3 impossible to guarantee. §15: one
clear file.

Order of operations, and why:

1. **Tier-0 estimate** (<15 ms) picks the band and therefore the policy.
2. **Prompt-side floor** runs before a single token is generated. An injection
   never reaches the model.
3. **Cache** is consulted after the floor, so a poisoned prompt cannot be served
   from cache either. A cache hit still passes the floor detectors on the way
   out — a cached response is not a trusted response.
4. **Route** using the features already computed.
5. **Schedule** the stream (gateway/scheduler.py).
6. **Regenerate** at most once, under a constraint prompt, and only with latency
   headroom left in the pack's budget.
7. **Record**: append to the hash-chained ledger, emit the OTel span, update the
   conversation ratchet.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from ..actions.masking import mask_entities
from ..actions.matrix import decide
from ..contracts import (
    Action,
    Band,
    DecisionRecord,
    GuardContext,
    RiskCategory,
    SignalMode,
    StreamPolicy,
    max_severity,
)
from ..guards.registry import DetectorRegistry, default_registry
from ..ledger.otel import emit_span
from ..ledger.store import Ledger
from ..policy.loader import PolicyLoader, get_loader
from ..risk.conversation import ConversationStore
from ..risk.estimator import EstimatorInput, RiskEstimator
from ..risk.features import ToolScope, hallucination_prior
from ..routing.cache import SemanticCache
from ..routing.router import Router, token_cost
from ..textutil import approx_tokens
from .providers.base import GenerationRequest, Provider
from .scheduler import Scheduler, StreamEvent, StreamResult, policy_for_arm

CONSTRAINT_SYSTEM = (
    "Answer strictly from the supplied context. Do not state any figure, name, "
    "date or identifier that does not appear verbatim in the context. If the "
    "context does not support an answer, say so."
)


@dataclass
class SlipstreamRequest:
    prompt: str
    use_case: str = "internal_knowledge"
    geography: str = "IN"
    retrieval_context: list[str] = field(default_factory=list)
    conversation_id: str | None = None
    tool_scope: ToolScope = ToolScope.NONE
    arm: str = "adaptive"
    fixture_id: str | None = None
    trace_id: str = ""
    band_override: Band | None = None  # tests only (INV-1 proof)
    use_cache: bool = True


@dataclass
class RunOutcome:
    record: DecisionRecord
    result: StreamResult
    band_reason: str = ""
    route_reason: str = ""
    cache_outcome: str = "miss"
    baseline_cost: float = 0.0
    actual_cost: float = 0.0


class SlipstreamGateway:
    def __init__(
        self,
        provider: Provider,
        *,
        registry: DetectorRegistry | None = None,
        loader: PolicyLoader | None = None,
        ledger: Ledger | None = None,
        cache: SemanticCache | None = None,
        conversations: ConversationStore | None = None,
        emit_telemetry: bool = True,
        backpressure_k: int = 3,
        enforce_budget: bool = True,
    ) -> None:
        self.provider = provider
        # INV-3: one registry instance, shared by every arm. Assertable by
        # identity in tests, not by reading the code.
        self.registry = registry or default_registry()
        self.loader = loader or get_loader()
        self.ledger = ledger
        self.cache = cache
        self.conversations = conversations or ConversationStore()
        self.router = Router()
        self.emit_telemetry = emit_telemetry
        self.backpressure_k = backpressure_k
        self.enforce_budget = enforce_budget
        self.estimator = RiskEstimator(self.loader)

    # -- main entry point ------------------------------------------------
    async def run(self, req: SlipstreamRequest) -> AsyncIterator[StreamEvent]:
        """Stream events, then stash the outcome on ``self.last_outcome``."""
        t0 = time.perf_counter()
        pack = self.loader.load(req.use_case)
        state = self.conversations.get(req.conversation_id)
        trace_id = req.trace_id or uuid.uuid4().hex

        est = self.estimator.estimate(
            EstimatorInput(
                prompt=req.prompt,
                use_case=req.use_case,
                geography=req.geography,
                retrieval_context=req.retrieval_context,
                tool_scope=req.tool_scope,
                conversation=state if req.conversation_id else None,
            )
        )
        policy = est.policy
        if req.band_override is not None:
            # Test hook for INV-1: force the cheapest possible policy and prove
            # the floor still holds.
            bc = pack.band_config(req.band_override)
            policy = StreamPolicy(
                band=req.band_override,
                holdback_tokens=bc.holdback_tokens,
                detectors=pack.detectors_for(req.band_override),
                sync=bc.sync,
                masking_enabled=pack.masking_enabled,
                escalation_armed=False,
                guard_budget_ms=bc.guard_budget_ms,
            )
        policy = policy_for_arm(req.arm, policy, pack)

        ctx = GuardContext(
            prompt=req.prompt,
            retrieval_context=req.retrieval_context,
            use_case=req.use_case,
            geography=req.geography,
            band=policy.band,
            pii_entities=pack.pii_entities(req.geography),
            protected_attributes=pack.bias.protected_attributes,
            decision_context=pack.bias.decision_context,
            capabilities=self.provider.capabilities(),
            budget_ms=policy.guard_budget_ms,
        )

        scheduler = Scheduler(
            self.provider,
            self.registry,
            pack,
            backpressure_k=self.backpressure_k,
            # Only the deliberately-leaky baseline is allowed to skip the inline
            # floor. Everything else, in every band, holds for it (INV-1).
            enforce_floor_inline=(req.arm != "fixed_async"),
            enforce_budget=self.enforce_budget,
        )
        result = StreamResult()

        # ---- 2. prompt-side floor (INV-1) -------------------------------
        prompt_ctx = ctx.model_copy(update={"is_prompt_side": True})
        t_pg = time.perf_counter()
        prompt_verdicts, prompt_decision = await scheduler.check_prompt(req.prompt, prompt_ctx)
        result.prompt_guard_ms = (time.perf_counter() - t_pg) * 1000.0
        result.verdicts.extend(prompt_verdicts)
        result.reasons.extend(f"prompt:{r}" for r in prompt_decision.reasons)
        # Injection is a prompt-side category; it belongs in the predicted set.
        result.triggered |= {
            c for c in prompt_decision.triggered if c is RiskCategory.INJECTION
        }

        if prompt_decision.action is Action.BLOCK:
            result.blocked = True
            result.action = Action.BLOCK
            result.total_latency_ms = (time.perf_counter() - t0) * 1000.0
            yield StreamEvent(
                type="blocked",
                text=prompt_decision.fallback_text or "",
                meta={"stage": "prompt", "reasons": prompt_decision.reasons},
            )
            yield StreamEvent(type="done", meta={"action": "block", "stage": "prompt"})
            self.last_outcome = self._finalise(
                req, pack, est, policy, result, trace_id, state,
                model_used="none", route_reason="blocked_before_generation",
                cache_outcome="skipped", elapsed=time.perf_counter() - t0,
            )
            return

        # ---- 3. cache ---------------------------------------------------
        cache_outcome = "disabled"
        cached_text: str | None = None
        if self.cache is not None and req.use_cache:
            cached_text, cache_outcome = self.cache.get(req.prompt, req.use_case)

        # ---- 4. route ---------------------------------------------------
        route = self.router.route(est.features, est.risk_vector, policy.band, pack)

        if cached_text is not None:
            async for ev in self._serve_cached(cached_text, policy, ctx, pack, result):
                yield ev
            result.total_latency_ms = (time.perf_counter() - t0) * 1000.0
            self.last_outcome = self._finalise(
                req, pack, est, policy, result, trace_id, state,
                model_used="cache", route_reason=route.reason,
                cache_outcome=cache_outcome, elapsed=time.perf_counter() - t0,
                baseline_model=route.model,
            )
            return

        # ---- 5. schedule ------------------------------------------------
        gen_req = GenerationRequest(
            prompt=req.prompt,
            model=route.model,
            fixture_id=req.fixture_id,
            metadata={"use_case": req.use_case},
        )
        async for ev in scheduler.stream(gen_req, policy, ctx, result):
            yield ev

        # ---- 6. regenerate (at most once, only with headroom) ------------
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        headroom = pack.latency_budget_ms - elapsed_ms
        if (
            result.action is Action.REGENERATE
            and pack.regenerate.enabled
            and headroom >= pack.regenerate.min_headroom_ms
        ):
            retry_req = GenerationRequest(
                prompt=req.prompt,
                system=CONSTRAINT_SYSTEM,
                model=route.model,
                fixture_id=req.fixture_id,
                metadata={"use_case": req.use_case, "variant": "constrained"},
            )
            retry = StreamResult()
            # `restart` tells the client to discard what it has rendered so far:
            # the retry replaces the response, it does not continue it. Without
            # this the two attempts concatenate on screen.
            yield StreamEvent(
                type="annotation",
                text="[regenerating under a grounding constraint]",
                meta={"attempt": 1, "restart": True},
            )
            async for ev in scheduler.stream(retry_req, policy, ctx, retry):
                yield ev
            retry.regenerated = True
            retry.verdicts = result.verdicts + retry.verdicts
            retry.reasons = result.reasons + ["regenerate:attempt1"] + retry.reasons
            retry.prompt_guard_ms = result.prompt_guard_ms
            # The first attempt's findings are part of what the system detected,
            # and its *emissions* already happened. A clean retry does not
            # retroactively un-leak a chunk the user already saw, and it does not
            # erase the masking that protected them. Both carry forward or the
            # safety metrics would credit the retry for the first attempt's work.
            retry.triggered |= result.triggered
            retry.abstained_categories |= result.abstained_categories
            retry.abstained = retry.abstained or result.abstained
            retry.guard_overhead_ms += result.guard_overhead_ms
            retry.leaked_chunks += result.leaked_chunks
            retry.leaked_entity_types = sorted(
                set(retry.leaked_entity_types) | set(result.leaked_entity_types)
            )
            retry.masked_before_emission = (
                retry.masked_before_emission or result.masked_before_emission
            )
            retry.entities_masked = sorted(
                set(retry.entities_masked) | set(result.entities_masked)
            )
            retry.degraded_mode = retry.degraded_mode or result.degraded_mode
            # What the system DID to this request was regenerate it. Recording
            # the clean retry's own ALLOW would report an intervention as a
            # no-op, and the action distribution would understate the work done.
            # A retry that came back worse still wins on severity.
            retry.action = max_severity([Action.REGENERATE, retry.action])
            # chunk_timings deliberately describe the retry stream only: the two
            # attempts have independent chunk indices and concatenating them
            # would corrupt the "had a successor" test in overlap_counts.
            retry.total_latency_ms = (time.perf_counter() - t0) * 1000.0
            result = retry
        elif result.action is Action.REGENERATE:
            result.action = Action.ESCALATE if pack.regulated else Action.ANNOTATE
            result.reasons.append(
                f"regenerate_no_headroom({headroom:.0f}ms<"
                f"{pack.regenerate.min_headroom_ms}ms)->downgraded"
            )
            if result.action is Action.ESCALATE:
                result.escalated = True

        result.total_latency_ms = (time.perf_counter() - t0) * 1000.0

        if self.cache is not None and req.use_cache and result.action is Action.ALLOW:
            self.cache.put(req.prompt, req.use_case, result.released_text)

        self.last_outcome = self._finalise(
            req, pack, est, policy, result, trace_id, state,
            model_used=route.model, route_reason=route.reason,
            cache_outcome=cache_outcome, elapsed=time.perf_counter() - t0,
        )

    async def collect(self, req: SlipstreamRequest) -> RunOutcome:
        """Drive the stream to completion and return the outcome. Used by the
        eval harness, which does not care about the events themselves."""
        async for _ in self.run(req):
            pass
        return self.last_outcome

    # -- helpers ---------------------------------------------------------
    async def _serve_cached(
        self, text: str, policy: StreamPolicy, ctx: GuardContext, pack, result: StreamResult
    ) -> AsyncIterator[StreamEvent]:
        """A cache hit still passes the floor. Cached is not the same as trusted."""
        detectors = self.registry.select([], side="output")
        verdicts = await self.registry.run(detectors, text, ctx)
        result.verdicts.extend(verdicts)
        d = decide(
            verdicts, pack, band=policy.band, verifiable=ctx.verifiable,
            masking_enabled=policy.masking_enabled,
        )
        out = text
        if d.entities_to_mask and policy.masking_enabled:
            out, types = mask_entities(text, d.entities_to_mask)
            result.entities_masked = types
            result.masked_before_emission = True
        result.reasons.extend(f"cache:{r}" for r in d.reasons)
        result.triggered |= d.triggered
        result.abstained_categories |= d.abstained
        result.abstained = result.abstained or bool(d.abstained)
        result.action = d.action
        result.text = text
        result.tokens_out = approx_tokens(text)
        if d.action is Action.BLOCK:
            result.blocked = True
            yield StreamEvent(type="blocked", text=d.fallback_text or "")
        else:
            result.released_text = out
            result.ttft_ms = 0.5
            yield StreamEvent(type="chunk", text=out, index=0, meta={"cached": True})
        yield StreamEvent(type="done", meta={"action": result.action.value, "cached": True})

    def _finalise(
        self,
        req: SlipstreamRequest,
        pack,
        est,
        policy: StreamPolicy,
        result: StreamResult,
        trace_id: str,
        state,
        *,
        model_used: str,
        route_reason: str,
        cache_outcome: str,
        elapsed: float,
        baseline_model: str | None = None,
    ) -> RunOutcome:
        # Output-side hallucination prior, recorded with the signal set that
        # produced it (§7.4). Reported separately per mode by the eval.
        prior, signal_mode = hallucination_prior(
            result.text or result.released_text,
            capabilities=self.provider.capabilities(),
            logprobs=None,
            retrieval_context=req.retrieval_context,
        )
        scores = dict(est.risk_vector.scores)
        scores[RiskCategory.HALLUCINATION] = round(
            max(scores.get(RiskCategory.HALLUCINATION, 0.0), prior * 0.6), 4
        )
        rv = est.risk_vector.model_copy(update={"scores": scores})

        tokens_in = approx_tokens(req.prompt) + sum(
            approx_tokens(c) for c in req.retrieval_context
        )
        record = DecisionRecord(
            record_id=uuid.uuid4().hex,
            trace_id=trace_id,
            use_case=req.use_case,
            geography=req.geography,
            conversation_id=req.conversation_id,
            turn_index=state.turn_index if state else 0,
            risk_vector=rv,
            policy=policy,
            verdicts=result.verdicts,
            action=result.action,
            entities_masked=result.entities_masked,
            degraded_mode=result.degraded_mode,
            abstained=result.abstained,
            regenerated=result.regenerated,
            escalated=result.escalated,
            signal_mode=signal_mode,
            ttft_ms=result.ttft_ms,
            total_latency_ms=result.total_latency_ms,
            guard_overhead_ms=result.guard_overhead_ms + result.prompt_guard_ms,
            tokens_in=tokens_in,
            tokens_out=result.tokens_out,
            model_used=model_used,
            arm=req.arm,
            chunk_timings=result.chunk_timings,
            cache_hit=cache_outcome in ("exact", "semantic"),
            cache_gated_miss=cache_outcome == "gated_miss",
            leaked_chunks=result.leaked_chunks,
        )

        if state is not None and req.conversation_id:
            state.observe(
                band=policy.band,
                risk=rv.max_score(),
                action=result.action,
                tool_scope=req.tool_scope,
                half_life_turns=pack.conversation.ratchet_half_life_turns,
            )

        if self.ledger is not None:
            record = self.ledger.append(record)
        if self.emit_telemetry:
            emit_span(record, pct_guard_first=result.pct_guard_finished_first)

        frontier = pack.routing.frontier_model
        baseline_cost = token_cost(baseline_model or frontier, tokens_in, record.tokens_out)
        actual_cost = (
            0.0 if record.cache_hit else token_cost(model_used, tokens_in, record.tokens_out)
        )

        return RunOutcome(
            record=record,
            result=result,
            band_reason=est.band_reason,
            route_reason=route_reason,
            cache_outcome=cache_outcome,
            baseline_cost=baseline_cost,
            actual_cost=actual_cost,
        )
