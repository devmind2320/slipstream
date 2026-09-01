"""Self-consistency. cost_class = expensive. HIGH band only.

Two modes, chosen by what the provider can actually do (§7.4):

* provider supports multi-sample -> semantic entropy over n resamples
* it does not                    -> intra-response self-contradiction only

If neither is usable the detector ABSTAINS. It never returns a low score to mean
"could not run"; that is the same mistake §9 warns about for groundedness.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from typing import Awaitable, Callable

from ..contracts import GuardContext, RiskCategory, Verdict
from ..embed import cosine, embed_array, semantic_entropy
from ..textutil import numbers, split_sentences
from .base import DetectorBase

SampleFn = Callable[[int], Awaitable[list[str]]]

# The sampler is per-request, but the detector registry is a process singleton so
# that INV-3 ("the three arms instantiate the identical detector registry") is
# assertable by identity. A context variable threads the request's sampler in
# without making the registry request-scoped.
current_sampler: ContextVar[SampleFn | None] = ContextVar("current_sampler", default=None)

_NEG = re.compile(r"\b(?:not|no|never|cannot|isn't|aren't|wasn't|doesn't|didn't|won't)\b", re.I)


def self_contradiction(text: str) -> tuple[float, list[str]]:
    """Sentence pairs that are near-duplicates but disagree on polarity or number."""
    sents = split_sentences(text)
    if len(sents) < 2:
        return 0.0, []
    vecs = [embed_array(s) for s in sents]
    evidence: list[str] = []
    worst = 0.0
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            sim = cosine(vecs[i], vecs[j])
            if sim < 0.60:
                continue
            neg_i = bool(_NEG.search(sents[i]))
            neg_j = bool(_NEG.search(sents[j]))
            n_i, n_j = numbers(sents[i]), numbers(sents[j])
            conflict = 0.0
            if neg_i != neg_j:
                conflict = max(conflict, 0.55 + 0.4 * (sim - 0.60) / 0.40)
            if n_i and n_j and n_i != n_j and (n_i & n_j) != max(n_i, n_j, key=len):
                if not (n_i <= n_j or n_j <= n_i):
                    conflict = max(conflict, 0.6 + 0.35 * (sim - 0.60) / 0.40)
            if conflict > worst:
                worst = conflict
                evidence = [f"sent{i}~sent{j} sim={sim:.2f}"]
    return min(1.0, worst), evidence


class ConsistencyDetector(DetectorBase):
    name = "consistency"
    category = RiskCategory.HALLUCINATION
    cost_class = "expensive"
    side = "output"

    def __init__(self, sample_fn: SampleFn | None = None, n_samples: int = 3) -> None:
        self.sample_fn = sample_fn
        self.n_samples = n_samples

    async def _check(self, text: str, ctx: GuardContext) -> Verdict:
        contra, ev = self_contradiction(text)
        sampler = self.sample_fn or current_sampler.get()

        if ctx.capabilities.supports_multi_sample and sampler is not None:
            samples = await sampler(self.n_samples)
            if len(samples) >= 2:
                ent = semantic_entropy([text, *samples])
                score = max(contra, ent)
                return self._verdict(
                    score,
                    evidence=f"semantic_entropy={ent:.2f} contradiction={contra:.2f}",
                )

        if len(split_sentences(text)) < 2:
            return self._verdict(
                0.0, evidence="no_multi_sample,single_sentence", abstained=True
            )
        return self._verdict(
            contra, evidence="self_contradiction_only " + " ".join(ev)
        )
