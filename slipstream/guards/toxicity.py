"""Toxicity. cost_class = cheap. Lexicon by default, per §4.

The lexicon is intentionally small and severity-weighted rather than a long
wordlist: a wordlist without severity weights produces the alert fatigue this
project is arguing against. Negation and quotation context discount a hit,
because "we do not tolerate slurs" is not a toxic utterance.

SLIPSTREAM_TOXICITY_BACKEND=bert swaps in unitary/toxic-bert.
"""

from __future__ import annotations

import re

from .. import settings
from ..contracts import GuardContext, RiskCategory, Verdict
from ..textutil import tokens
from .base import DetectorBase

# term -> severity. Severe slurs are not enumerated here; the shipped lexicon
# covers harassment/insult/threat registers, which is what the synthetic corpus
# exercises. A production deployment would swap in a maintained list.
LEXICON: dict[str, float] = {
    "idiot": 0.55, "idiots": 0.55, "stupid": 0.45, "moron": 0.6, "morons": 0.6,
    "worthless": 0.6, "useless": 0.4, "incompetent": 0.4, "pathetic": 0.55,
    "scum": 0.75, "trash": 0.5, "garbage": 0.45, "disgusting": 0.5,
    "hate": 0.5, "kill": 0.8, "die": 0.65, "destroy": 0.4, "attack": 0.4,
    "shut up": 0.6, "screw you": 0.75, "go away": 0.3,
    "loser": 0.55, "losers": 0.55, "freak": 0.6, "dumb": 0.5, "dumbass": 0.8,
    "threat": 0.35, "revenge": 0.5, "punish": 0.4,
    "inferior": 0.6, "subhuman": 0.9, "vermin": 0.85, "parasite": 0.7,
}

_NEGATION = re.compile(
    r"\b(?:not|never|no|don't|do not|doesn't|won't|cannot|can't|refuse to|"
    r"we do not|policy against|prohibited|unacceptable)\b",
    re.I,
)
_QUOTED = re.compile(r"[\"“'‘][^\"”'’]{0,120}[\"”'’]")

_bert = None


def _load_bert():
    global _bert
    if _bert is None:
        from transformers import pipeline  # lazy, optional

        _bert = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
    return _bert


def lexicon_score(text: str) -> tuple[float, list[str]]:
    low = text.lower()
    toks = set(tokens(text))
    hits: list[tuple[str, float]] = []
    for term, sev in LEXICON.items():
        if " " in term:
            if term in low:
                hits.append((term, sev))
        elif term in toks:
            hits.append((term, sev))
    if not hits:
        return 0.0, []

    score = max(s for _, s in hits)
    # More than one distinct hit compounds, but sub-linearly.
    if len(hits) > 1:
        score = min(1.0, score + 0.08 * (len(hits) - 1))

    quoted_spans = _QUOTED.findall(text)
    if any(any(t in q.lower() for t, _ in hits) for q in quoted_spans):
        score *= 0.5  # reported speech
    for term, _ in hits:
        idx = low.find(term)
        window = low[max(0, idx - 60) : idx]
        if _NEGATION.search(window):
            score *= 0.45
            break
    return min(1.0, score), [t for t, _ in sorted(hits, key=lambda h: -h[1])[:5]]


class ToxicityDetector(DetectorBase):
    name = "toxicity"
    category = RiskCategory.TOXICITY
    cost_class = "cheap"
    side = "both"

    async def _check(self, text: str, ctx: GuardContext) -> Verdict:
        if settings.TOXICITY_BACKEND == "bert":
            try:
                out = _load_bert()(text[:1000])[0]
                score = max(float(d["score"]) for d in out if d["label"] != "neutral")
                return self._verdict(score, evidence="toxic-bert")
            except Exception:  # noqa: BLE001 - fall back rather than fail open
                pass
        score, hits = lexicon_score(text)
        return self._verdict(score, evidence=("terms=" + ",".join(hits)) if hits else None)
