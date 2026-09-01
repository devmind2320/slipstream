"""Model tier routing (§7.10).

The complexity classifier reuses the Tier-0 features that were already computed
for band selection — no extra inference pass, no second model. That is the only
way routing can pay for itself at this latency budget.

Two guardrails on the routing decision:
  * a HIGH-band request never routes to the cheap tier, whatever its complexity.
    Saving tokens on the traffic most likely to cause harm is the wrong trade.
  * a request whose risk vector shows a verifiability gap routes up, because a
    cheap model with nothing to check against is where fabrication lives.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import Band, RiskCategory, RiskVector
from ..policy.loader import PolicyPack
from ..risk.features import PromptFeatures

# Illustrative $/1k tokens. Ratios matter, absolute values do not; the eval
# reports a token-cost *delta*, and the ratio is what drives it.
TOKEN_PRICES: dict[str, float] = {
    "replay-cheap": 0.00015,
    "replay-frontier": 0.00250,
    "replay-sim": 0.00100,
}


@dataclass
class RouteDecision:
    model: str
    tier: str  # "cheap" | "frontier"
    complexity: float
    reason: str


class Router:
    def route(
        self,
        features: PromptFeatures,
        risk: RiskVector,
        band: Band,
        pack: PolicyPack,
    ) -> RouteDecision:
        cfg = pack.routing
        complexity = features.complexity

        if band is Band.HIGH:
            return RouteDecision(
                cfg.frontier_model, "frontier", complexity, "high_band_never_cheap"
            )
        if (
            not risk.verifiable
            and risk.scores.get(RiskCategory.HALLUCINATION, 0.0) >= 0.45
        ):
            return RouteDecision(
                cfg.frontier_model, "frontier", complexity, "unverifiable_and_risky"
            )
        if complexity >= cfg.complexity_threshold:
            return RouteDecision(
                cfg.frontier_model, "frontier", complexity, "complexity>=threshold"
            )
        return RouteDecision(cfg.cheap_model, "cheap", complexity, "cheap_tier_sufficient")


def token_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price = TOKEN_PRICES.get(model, TOKEN_PRICES["replay-sim"])
    return price * (tokens_in + tokens_out) / 1000.0
