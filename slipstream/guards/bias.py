"""Policy-scoped bias check. cost_class = cheap.

Deliberately NOT a general-purpose bias classifier (§7.5). We would not build one
well at this scale and a reviewer would spot it. What this does instead is narrow
and defensible:

    The policy pack declares a set of protected attributes and whether this use
    case is a decision context. The detector flags when a protected attribute
    surfaces inside a recommendation or a decision rationale.

That catches the concrete failure enterprises care about — "we declined the
application because the applicant is a new mother" — and stays silent on the
open-ended question of whether a passage is "biased". The limitation is stated in
the README rather than hidden.

Outside a decision context this returns a clean 0.0, not an abstention: "no
decision was made here, so no biased decision was made here" is a real answer,
and folding it into the abstention rate would corrupt that metric.
"""

from __future__ import annotations

import re

from ..contracts import GuardContext, RiskCategory, Verdict
from ..textutil import split_sentences
from .base import DetectorBase

ATTRIBUTE_TERMS: dict[str, tuple[str, ...]] = {
    "gender": (
        "she", "her", "he", "his", "him", "female", "male", "woman", "women",
        "man", "men", "mother", "father", "maternity", "paternity", "pregnant",
        "pregnancy", "transgender", "nonbinary",
    ),
    "age": (
        "young", "younger", "old", "older", "elderly", "aged", "millennial",
        "boomer", "fresh graduate", "near retirement", "over 50", "under 30",
        "senior citizen", "age",
    ),
    "ethnicity": (
        "ethnic", "ethnicity", "race", "racial", "caste", "tribal", "indigenous",
        "immigrant", "foreign national", "regional background", "surname suggests",
        "nationality",
    ),
    "disability": (
        "disabled", "disability", "wheelchair", "blind", "deaf", "chronic illness",
        "mental health", "neurodivergent", "medical condition", "sick leave",
    ),
    "marital_status": (
        "married", "unmarried", "single", "divorced", "widowed", "spouse",
        "family commitments", "dependents", "childcare",
    ),
    "religion": ("religion", "religious", "faith", "hindu", "muslim", "christian", "sikh"),
}

DECISION_MARKERS = re.compile(
    r"\b(?:recommend(?:ed|ation)?|approv\w+|reject\w+|declin\w+|den(?:y|ied|ial)|"
    r"reduc\w+\s+(?:the\s+)?(?:limit|score|offer)|shortlist\w*|hire|hiring|"
    r"not\s+(?:a\s+)?(?:suitable|good\s+fit|eligible)|eligib\w+|ineligible|"
    r"promot(?:e|ion|ed)|terminat\w+|score\s+(?:of|was)|risk\s+(?:tier|rating|band)|"
    r"prioriti[sz]\w+|disqualif\w+|flag(?:ged)?\s+for\s+review|offer\s+(?:should|was))\b",
    re.I,
)

RATIONALE_MARKERS = re.compile(
    r"\b(?:because|since|due\s+to|owing\s+to|given\s+(?:that|his|her|their)|"
    r"as\s+(?:she|he|they)|on\s+account\s+of|reflect(?:s|ing)\s+(?:her|his|their)|"
    r"considering|factor(?:ed|ing)?\s+in|takes?\s+into\s+account)\b",
    re.I,
)


def _attribute_hits(sentence: str, attributes: list[str]) -> list[str]:
    low = " " + sentence.lower() + " "
    found: list[str] = []
    for attr in attributes:
        for term in ATTRIBUTE_TERMS.get(attr, ()):
            needle = f" {term} " if " " not in term else term
            if needle in low:
                found.append(f"{attr}:{term}")
                break
    return found


class BiasDetector(DetectorBase):
    name = "bias"
    category = RiskCategory.BIAS
    cost_class = "cheap"
    side = "output"

    async def _check(self, text: str, ctx: GuardContext) -> Verdict:
        if not ctx.decision_context or not ctx.protected_attributes:
            return self._verdict(0.0, evidence="out_of_scope:not_a_decision_context")

        best = 0.0
        evidence: list[str] = []
        for i, sent in enumerate(split_sentences(text) or [text]):
            attrs = _attribute_hits(sent, ctx.protected_attributes)
            if not attrs:
                continue
            decision = DECISION_MARKERS.search(sent)
            if not decision:
                continue
            rationale = RATIONALE_MARKERS.search(sent)
            # Attribute + decision in one sentence is suspicious; add an explicit
            # causal connector and it is a stated rationale, which is the case
            # the pack actually cares about.
            score = 0.88 if rationale else 0.62
            if score > best:
                best = score
                evidence = [
                    f"sent{i} attrs={'|'.join(attrs)} "
                    f"decision={decision.group(0)!r} "
                    f"rationale={'yes' if rationale else 'no'}"
                ]
        return self._verdict(best, evidence=" ".join(evidence) if evidence else None)
