"""Detector protocol and shared plumbing.

``cost_class`` is what the scheduler and the policy packs reason about:

* ``floor``     — runs on prompt AND every output chunk in every band, always.
                  INV-1 lives here. Never budget-skipped, never degraded away.
* ``cheap``     — low tens of ms; runs in MEDIUM and above.
* ``expensive`` — HIGH band only, and only when the provider supports what it
                  needs.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal, Protocol, runtime_checkable

from .. import settings
from ..contracts import GuardContext, RiskCategory, Verdict

CostClass = Literal["floor", "cheap", "expensive"]


# Abstentions that mean "there was nothing to run against" cost nothing in a real
# deployment either — the NLI model is never invoked. Flooring those at the model
# latency would charge unverifiable traffic for a call that never happens.
NO_WORK_EVIDENCE = frozenset({"no_grounding_source", "empty_context"})


async def _apply_latency_model(cost_class: CostClass, elapsed_ms: float) -> float:
    """Floor a detector's wall-clock at the §4 CPU target for its class.

    See settings.LATENCY_PROFILE for why. The verdict is always the real
    engine's; this only makes the *cost* of verification representative of the
    transformer stack the offline engines substitute for, so the three-arm
    latency comparison measures something real instead of measuring noise.
    """
    if settings.LATENCY_PROFILE != "spec":
        return elapsed_ms
    target = settings.LATENCY_TARGETS_MS.get(cost_class, 0.0)
    if elapsed_ms >= target:
        return elapsed_ms
    await asyncio.sleep((target - elapsed_ms) / 1000.0)
    return target


@runtime_checkable
class Detector(Protocol):
    name: str
    category: RiskCategory
    cost_class: CostClass

    async def check(self, text: str, ctx: GuardContext) -> Verdict: ...


class DetectorBase:
    """Base with timing built in — §15: log timing everywhere from the start."""

    name: str = "base"
    category: RiskCategory = RiskCategory.HALLUCINATION
    cost_class: CostClass = "cheap"
    # Detectors that only make sense on the way in (injection) or on the way
    # out (pii, groundedness). "both" means it runs on either side.
    side: Literal["prompt", "output", "both"] = "output"

    async def check(self, text: str, ctx: GuardContext) -> Verdict:
        t0 = time.perf_counter()
        try:
            v = await self._check(text, ctx)
        except Exception as exc:  # noqa: BLE001
            # A detector crash must not take the stream down, and must not be
            # silently read as "clean". Abstain and let the action matrix decide.
            elapsed = await _apply_latency_model(
                self.cost_class, (time.perf_counter() - t0) * 1000.0
            )
            return Verdict(
                detector=self.name,
                category=self.category,
                score=0.0,
                latency_ms=elapsed,
                evidence=f"detector_error:{type(exc).__name__}",
                abstained=True,
            )
        raw_ms = (time.perf_counter() - t0) * 1000.0
        if v.abstained and v.evidence in NO_WORK_EVIDENCE:
            return v.model_copy(update={"latency_ms": raw_ms})
        elapsed = await _apply_latency_model(self.cost_class, raw_ms)
        return v.model_copy(update={"latency_ms": elapsed})

    async def _check(self, text: str, ctx: GuardContext) -> Verdict:  # pragma: no cover
        raise NotImplementedError

    def _verdict(
        self,
        score: float,
        *,
        evidence: str | None = None,
        abstained: bool = False,
        entities: list | None = None,
        category: RiskCategory | None = None,
    ) -> Verdict:
        return Verdict(
            detector=self.name,
            category=category or self.category,
            score=max(0.0, min(1.0, score)),
            latency_ms=0.0,
            evidence=evidence,
            abstained=abstained,
            entities=entities or [],
        )
