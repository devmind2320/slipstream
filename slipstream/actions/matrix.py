"""Action matrix (§8).

Two properties this file must have:

1. **Max severity across triggered categories, not first match** (§9). Every rule
   that fires contributes a candidate action; the outcome is the most severe of
   them. A first-match implementation would let an ANNOTATE-worthy hallucination
   mask a BLOCK-worthy injection purely because of rule ordering.

2. **Abstention is routed, not scored.** A groundedness abstention means "no
   source to check against". In a regulated pack that ESCALATEs; elsewhere it
   ANNOTATEs with an explicit unverifiable notice. It never counts as clean and
   it never counts as a violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts import (
    Action,
    Band,
    EntityHit,
    RiskCategory,
    Verdict,
    max_severity,
)
from ..policy.loader import PolicyPack

UNVERIFIABLE_NOTICE = (
    "[unverifiable: no grounding source was attached to this request, so the "
    "claims above were not checked against a source]"
)
UNCERTAIN_NOTICE = (
    "[uncertain: this response scored close to the review threshold and has been "
    "logged for human review]"
)
TOXICITY_FALLBACK = (
    "This response was withheld because it did not meet content policy. "
    "Please rephrase, or ask for a human agent."
)
INJECTION_FALLBACK = (
    "This request was blocked: it contains instructions that attempt to override "
    "the assistant's operating policy."
)


@dataclass
class ActionDecision:
    action: Action = Action.ALLOW
    reasons: list[str] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)
    entities_to_mask: list[EntityHit] = field(default_factory=list)
    entity_types: list[str] = field(default_factory=list)
    triggered: set[RiskCategory] = field(default_factory=set)
    abstained: set[RiskCategory] = field(default_factory=set)
    near_threshold: bool = False
    fallback_text: str | None = None
    feedback_worthy: bool = False

    @property
    def blocks_release(self) -> bool:
        return self.action in (Action.BLOCK, Action.ESCALATE, Action.REGENERATE)


def decide(
    verdicts: list[Verdict],
    pack: PolicyPack,
    *,
    band: Band = Band.MEDIUM,
    verifiable: bool = False,
    masking_enabled: bool = True,
    is_prompt_side: bool = False,
) -> ActionDecision:
    d = ActionDecision()
    candidates: list[Action] = [Action.ALLOW]
    margin = pack.uncertainty_margin

    for v in verdicts:
        thr = pack.threshold(v.category)

        if v.abstained:
            d.abstained.add(v.category)
            if v.category is RiskCategory.HALLUCINATION:
                # §9: "I cannot check this" != "this is wrong".
                if pack.regulated:
                    candidates.append(Action.ESCALATE)
                    d.reasons.append(f"{v.detector}:abstain->escalate(regulated)")
                else:
                    candidates.append(Action.ANNOTATE)
                    d.annotations.append(UNVERIFIABLE_NOTICE)
                    d.reasons.append(f"{v.detector}:abstain->annotate")
            else:
                d.reasons.append(f"{v.detector}:abstain(no_action)")
            continue

        if v.score >= thr:
            d.triggered.add(v.category)

            if v.category is RiskCategory.INJECTION:
                candidates.append(Action.BLOCK)
                d.fallback_text = INJECTION_FALLBACK
                d.reasons.append(f"injection:{v.score:.2f}>={thr:.2f}->block")

            elif v.category is RiskCategory.PRIVACY:
                if is_prompt_side:
                    # Inbound PII is the user's own data; it is logged and the
                    # band is already lifted, but it is not the leak path.
                    candidates.append(Action.ANNOTATE)
                    d.reasons.append(f"pii:inbound:{v.score:.2f}->annotate")
                elif masking_enabled:
                    candidates.append(Action.MASK)
                    d.entities_to_mask.extend(v.entities)
                    d.entity_types = sorted(
                        {e.entity_type for e in d.entities_to_mask}
                    )
                    d.reasons.append(
                        f"pii:{v.score:.2f}>={thr:.2f}->mask({','.join(d.entity_types)})"
                    )
                elif pack.require_masking:
                    candidates.append(Action.BLOCK)
                    d.reasons.append("pii:masking_disabled_but_required->block")
                else:
                    candidates.append(Action.ANNOTATE)
                    d.reasons.append("pii:masking_disabled->annotate")

            elif v.category is RiskCategory.HALLUCINATION:
                if verifiable and pack.regulated:
                    candidates.append(Action.ESCALATE)
                    d.reasons.append(
                        f"groundedness:{v.score:.2f}>={thr:.2f}->escalate(regulated)"
                    )
                elif verifiable:
                    candidates.append(Action.REGENERATE)
                    d.reasons.append(
                        f"groundedness:{v.score:.2f}>={thr:.2f}->regenerate"
                    )
                else:
                    candidates.append(Action.ANNOTATE)
                    d.annotations.append(UNVERIFIABLE_NOTICE)
                    d.reasons.append("groundedness:unverifiable->annotate")

            elif v.category is RiskCategory.TOXICITY:
                candidates.append(Action.BLOCK)
                d.fallback_text = d.fallback_text or TOXICITY_FALLBACK
                d.reasons.append(f"toxicity:{v.score:.2f}>={thr:.2f}->block")

            elif v.category is RiskCategory.BIAS:
                if pack.bias.decision_context:
                    candidates.append(Action.ESCALATE)
                    d.reasons.append(f"bias:{v.score:.2f}>={thr:.2f}->escalate")
                else:
                    candidates.append(Action.ANNOTATE)
                    d.reasons.append(f"bias:{v.score:.2f}->annotate")

        elif thr - margin <= v.score < thr:
            # Inside the uncertainty margin: annotate and add to the feedback set
            # so the next calibration has a labelled example where it matters.
            d.near_threshold = True
            d.feedback_worthy = True
            candidates.append(Action.ANNOTATE)
            d.annotations.append(UNCERTAIN_NOTICE)
            d.reasons.append(
                f"{v.detector}:{v.score:.2f} within margin of {thr:.2f}->annotate"
            )

    d.action = max_severity(candidates)

    # REGENERATE only survives if the pack allows it and there is headroom; the
    # caller downgrades it when there is not (see scheduler).
    if d.action is Action.REGENERATE and not pack.regenerate.enabled:
        d.action = Action.ESCALATE if pack.regulated else Action.ANNOTATE
        d.reasons.append("regenerate_disabled->downgraded")

    # Escalation in a band that is not armed for it degrades to annotate, so a
    # LOW-band consumer pack cannot silently start paging humans.
    if d.action is Action.ESCALATE and band is Band.LOW and not pack.regulated:
        d.action = Action.ANNOTATE
        d.reasons.append("escalation_not_armed_in_low->annotate")

    return d
