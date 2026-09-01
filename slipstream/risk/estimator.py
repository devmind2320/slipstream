"""Tier-0 prompt-side risk estimation and policy selection (§7.2).

Budget: under 15 ms on CPU. Everything here is regex, dict lookups and one cached
embedding. **No LLM call** — an LLM in this path costs more than the verification
it is supposed to be scheduling, which would defeat the entire design.

INV-2 is enforced in ``_select_band``: when the score sits within the pack's
``uncertainty_margin`` of a band boundary, or when the estimator's self-confidence
is low, the band rounds **up**. Never down.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..contracts import (
    Band,
    BAND_ORDER,
    RiskCategory,
    RiskVector,
    StreamPolicy,
    max_band,
)
from ..policy.loader import PolicyPack
from .conversation import ConversationState, apply_ratchet
from .features import PromptFeatures, ToolScope, extract_features


@dataclass
class EstimatorInput:
    prompt: str
    use_case: str = "internal_knowledge"
    geography: str = "IN"
    retrieval_context: list[str] = field(default_factory=list)
    tool_scope: ToolScope = ToolScope.NONE
    conversation: ConversationState | None = None


@dataclass
class EstimatorResult:
    risk_vector: RiskVector
    policy: StreamPolicy
    features: PromptFeatures
    band_reason: str
    elapsed_ms: float


class RiskEstimator:
    def __init__(self, pack_provider=None) -> None:
        # Injected so the gateway, the eval harness and the tests all share one
        # loader instance and therefore one hot-reload cache.
        from ..policy.loader import get_loader

        self._loader = pack_provider or get_loader()

    # -- scoring ---------------------------------------------------------
    def _score(self, f: PromptFeatures, pack: PolicyPack) -> dict[RiskCategory, float]:
        """Multi-label. Categories overlap by design — a prompt asking for a
        fabricated detail about a real person raises both PRIVACY and
        HALLUCINATION, and we do not pick a winner."""
        topical = max(f.topic_score, f.risky_similarity)

        privacy = max(
            f.pii_score,
            0.55 * topical if f.pii_types else 0.35 * topical,
        )
        # Asking about a person while a retrieval source is attached is lower
        # risk than asking about a person with nothing to check against.
        if f.pii_types and not f.verifiable:
            privacy = min(1.0, privacy + 0.10)

        hallucination = 0.55 * topical + 0.25 * f.complexity
        if not f.verifiable:
            hallucination = min(1.0, hallucination + 0.18)
        else:
            hallucination = max(0.0, hallucination - 0.05)

        injection = f.injection_score
        toxicity = f.toxicity_prior

        bias = 0.0
        if pack.bias.decision_context and pack.bias.protected_attributes:
            bias = min(1.0, 0.40 + 0.45 * topical)

        scores = {
            RiskCategory.PRIVACY: privacy,
            RiskCategory.HALLUCINATION: hallucination,
            RiskCategory.INJECTION: injection,
            RiskCategory.TOXICITY: toxicity,
            RiskCategory.BIAS: bias,
        }

        # Tool side-effects and accumulated conversation risk lift every
        # category: the cost of being wrong has gone up across the board.
        lift = max(f.tool_risk, 0.6 * f.conversation_risk)
        if lift > 0:
            scores = {k: min(1.0, v + lift * (1.0 - v)) for k, v in scores.items()}
        return {k: round(min(1.0, max(0.0, v)), 4) for k, v in scores.items()}

    def _confidence(self, f: PromptFeatures, scores: dict[RiskCategory, float]) -> float:
        """How much the estimator trusts its own read.

        Short prompts, no corroborating signals, and scores clustered in the
        middle all reduce confidence — and low confidence rounds the band up.
        """
        conf = 0.55
        if f.prompt_tokens >= 12:
            conf += 0.15
        if f.topic_hits:
            conf += 0.12
        if f.pii_types:
            conf += 0.10
        if f.injection_score >= 0.8:
            conf += 0.15
        top = max(scores.values(), default=0.0)
        # Decisive scores (very low or very high) are easier to trust.
        conf += 0.15 * abs(top - 0.5) * 2.0
        return round(max(0.05, min(1.0, conf)), 4)

    # -- band selection --------------------------------------------------
    def _select_band(
        self, top_score: float, confidence: float, pack: PolicyPack
    ) -> tuple[Band, str]:
        med = pack.band_thresholds.get(Band.MEDIUM, 0.35)
        high = pack.band_thresholds.get(Band.HIGH, 0.65)
        margin = pack.uncertainty_margin

        if top_score >= high:
            band, reason = Band.HIGH, "score>=high"
        elif top_score >= med:
            band, reason = Band.MEDIUM, "score>=medium"
        else:
            band, reason = Band.LOW, "score<medium"

        # INV-2: inside the margin *below* a boundary, round up. Never down.
        if band is not Band.HIGH and high - margin <= top_score < high:
            band, reason = Band.HIGH, f"INV2:within_margin_of_high({margin})"
        elif band is Band.LOW and med - margin <= top_score < med:
            band, reason = Band.MEDIUM, f"INV2:within_margin_of_medium({margin})"

        # INV-2 again: an estimator that does not trust itself must not be the
        # thing that chooses the cheaper policy.
        if confidence < 0.5 and band is not Band.HIGH:
            band = Band(
                {Band.LOW: Band.MEDIUM, Band.MEDIUM: Band.HIGH}[band]
            )
            reason = f"INV2:low_confidence({confidence:.2f})"

        floor = pack.effective_min_band()
        if BAND_ORDER[floor] > BAND_ORDER[band]:
            band, reason = floor, f"pack_min_band={floor.value}"
        return band, reason

    # -- entry point -----------------------------------------------------
    def estimate(self, req: EstimatorInput) -> EstimatorResult:
        t0 = time.perf_counter()
        pack = self._loader.load(req.use_case)
        state = req.conversation

        f = extract_features(
            req.prompt,
            use_case=req.use_case,
            regulated_topics=pack.regulated_topics,
            risky_prompt_examples=pack.risky_prompt_examples,
            retrieval_context=req.retrieval_context,
            tool_scope=req.tool_scope,
            conversation_risk=state.accumulated_risk if state else 0.0,
        )

        scores = self._score(f, pack)
        confidence = self._confidence(f, scores)
        top = max(scores.values(), default=0.0)

        band, reason = self._select_band(top, confidence, pack)

        if state is not None:
            band, ratchet_reason = apply_ratchet(
                band,
                state,
                half_life_turns=pack.conversation.ratchet_half_life_turns,
                min_band=pack.effective_min_band(),
                tool_scope=req.tool_scope,
                tool_write_forces_high=pack.conversation.tool_write_forces_high,
            )
            if ratchet_reason != "turn_band":
                reason = f"{reason}|{ratchet_reason}"
        elif req.tool_scope in (ToolScope.WRITE, ToolScope.IRREVERSIBLE):
            band = max_band(band, Band.HIGH)
            reason = f"{reason}|tool_scope={req.tool_scope.value}_forces_high"

        bc = pack.band_config(band)
        policy = StreamPolicy(
            band=band,
            holdback_tokens=bc.holdback_tokens,
            detectors=pack.detectors_for(band),
            sync=bc.sync,
            masking_enabled=pack.masking_enabled,
            escalation_armed=band is not Band.LOW or pack.regulated,
            guard_budget_ms=bc.guard_budget_ms,
        )

        rv = RiskVector(scores=scores, verifiable=f.verifiable, confidence=confidence)
        return EstimatorResult(
            risk_vector=rv,
            policy=policy,
            features=f,
            band_reason=reason,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )
