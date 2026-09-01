"""Detector registry.

INV-3 depends on this file: the three evaluation arms differ only in scheduling
policy, so they must be handed detector instances that are provably the same.
``fingerprint()`` makes that assertable in a test rather than a claim in a README.

The registry also owns two rules the scheduler must not be able to violate:

* floor detectors are ALWAYS in the selected set, in every band, and are never
  dropped by degraded mode (INV-1);
* the guard budget can time out a cheap or expensive detector, but never a floor
  one. A timed-out detector abstains — it never returns a clean score.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Iterable, Sequence

from ..contracts import GuardContext, Verdict
from .base import Detector
from .bias import BiasDetector
from .consistency import ConsistencyDetector
from .groundedness import GroundednessDetector
from .injection import InjectionDetector
from .pii import PIIDetector
from .toxicity import ToxicityDetector


class DetectorRegistry:
    def __init__(self, detectors: Iterable[Detector]) -> None:
        self._by_name: dict[str, Detector] = {d.name: d for d in detectors}

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)

    @property
    def names(self) -> list[str]:
        return sorted(self._by_name)

    def get(self, name: str) -> Detector:
        return self._by_name[name]

    def floor(self) -> list[Detector]:
        return [d for _, d in sorted(self._by_name.items()) if d.cost_class == "floor"]

    def fingerprint(self) -> str:
        """Stable digest over detector identity. Two registries with the same
        fingerprint enforce the same thing (INV-3)."""
        parts = [
            f"{d.name}:{type(d).__module__}.{type(d).__qualname__}:"
            f"{d.category.value}:{d.cost_class}"
            for _, d in sorted(self._by_name.items())
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    # -- selection -------------------------------------------------------
    def select(
        self,
        requested: Sequence[str],
        *,
        side: str = "output",
        degraded: bool = False,
    ) -> list[Detector]:
        """Floor first, always. ``degraded`` drops everything except the floor."""
        chosen: dict[str, Detector] = {d.name: d for d in self.floor()}
        if not degraded:
            for name in requested:
                d = self._by_name.get(name)
                if d is not None:
                    chosen[name] = d
        out = [
            d
            for _, d in sorted(chosen.items())
            if getattr(d, "side", "both") in ("both", side)
        ]
        return out

    # -- execution -------------------------------------------------------
    async def run(
        self,
        detectors: Sequence[Detector],
        text: str,
        ctx: GuardContext,
        *,
        budget_ms: int | None = None,
    ) -> list[Verdict]:
        """Run detectors concurrently. Floor detectors ignore the budget."""
        if not detectors:
            return []
        budget_s = (budget_ms / 1000.0) if budget_ms else None

        async def _one(d: Detector) -> Verdict:
            if budget_s is None or d.cost_class == "floor":
                return await d.check(text, ctx)
            try:
                return await asyncio.wait_for(d.check(text, ctx), timeout=budget_s)
            except asyncio.TimeoutError:
                return Verdict(
                    detector=d.name,
                    category=d.category,
                    score=0.0,
                    latency_ms=budget_ms or 0.0,
                    evidence="budget_exceeded",
                    abstained=True,
                )

        return list(await asyncio.gather(*(_one(d) for d in detectors)))


def build_registry(sample_fn=None) -> DetectorRegistry:
    """The one place detectors are constructed. All arms call this."""
    return DetectorRegistry(
        [
            PIIDetector(),
            InjectionDetector(),
            GroundednessDetector(),
            ToxicityDetector(),
            ConsistencyDetector(sample_fn=sample_fn),
            BiasDetector(),
        ]
    )


_default: DetectorRegistry | None = None


def default_registry() -> DetectorRegistry:
    global _default
    if _default is None:
        _default = build_registry()
    return _default
