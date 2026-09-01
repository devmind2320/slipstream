"""Groundedness. cost_class = cheap.

This is where §9's hardest point is enforced:

    **If no retrieval context is attached, this detector MUST abstain.**

Not "score low". Abstention is a third state, distinct from clean and dirty, and
it routes differently (ANNOTATE in low-stakes packs, ESCALATE in regulated ones).
A system that reads "I cannot check this" as "this is false" produces exactly the
alert fatigue the brief warns about.

Default engine (offline) is a lexical entailment proxy per claim sentence:
  * content-token coverage against the best-matching context sentence
  * numeric agreement — a figure in the claim that appears nowhere in the context
    is the strongest fabrication signal we have and is weighted accordingly
  * entity agreement — capitalised tokens absent from context
  * a hedging discount, since hedged claims assert less

Backend "nli" swaps in cross-encoder/nli-deberta-v3-small for the per-claim
entailment score and leaves the rest of the pipeline identical.
"""

from __future__ import annotations

import re

from .. import settings
from ..contracts import GuardContext, RiskCategory, Verdict
from ..embed import cosine, embed_array
from ..textutil import (
    STOPWORDS,
    content_tokens,
    hedging_ratio,
    numbers,
    split_sentences,
)
from .base import DetectorBase

_nli_model = None


def _load_nli():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder  # lazy, optional

        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _nli_model


def _named_entities(claim: str) -> set[str]:
    """Capitalised tokens that plausibly name something.

    Sentence-initial words are excluded: English capitalises them for grammar,
    not for reference, and counting "Under" or "However" as an unsupported entity
    was flagging correctly-grounded prose as fabricated.
    """
    words = claim.split()
    out: set[str] = set()
    for i, w in enumerate(words):
        bare = w.strip(".,;:()[]\"'")
        if len(bare) <= 2 or not bare[:1].isupper():
            continue
        if i == 0 or (words[i - 1].strip()[-1:] in ".!?"):
            continue  # sentence-initial: grammatical capital, not a name
        if bare.lower() in STOPWORDS:
            continue
        out.add(bare)
    return out


# Sentences whose subject is the assistant or the user and which assert nothing
# checkable — "I have noted your request", "Let me know if you need anything".
# Entailment is undefined for these, and scoring them against a source document
# is how a groundedness detector generates the alert fatigue §9 warns about.
_PROCESS_OPENER = re.compile(
    r"^\s*(?:I|We|You|Let\s+me|Please|Thanks|Thank\s+you|Is\s+there|Could\s+you|"
    r"Feel\s+free|Happy\s+to|Sorry|Apologies|Hope\s+)\b",
    re.I,
)


def _is_checkable(claim: str) -> bool:
    """Claim extraction before entailment.

    Groundedness is a property of assertions about the source, not of every
    sentence in the response. A sentence qualifies if it carries a figure, names
    something, or says enough to be checkable at all.
    """
    if len(content_tokens(claim)) < 3:
        return False
    grounded_signal = bool(numbers(claim)) or bool(_named_entities(claim))
    if _PROCESS_OPENER.match(claim) and not grounded_signal:
        return False
    return True


def _lexical_support(claim: str, context_sentences: list[str]) -> tuple[float, float]:
    """Returns (support, confidence).

    **support** — 0.0 wholly unsupported, 1.0 fully entailed.

    The premise set includes each context sentence *and* the whole context
    concatenated. Source documents get split into short sentences ("Debt-to-income
    ratio 18%."), and a correct summary that aggregates four of them entails from
    none of them individually. Scoring against single sentences alone flagged
    correct multi-fact summaries as fabrications — measured, then fixed.

    **confidence** — how much this claim's shortfall should be believed. A figure
    or a name in the claim that appears nowhere in the source is hard evidence of
    fabrication. Prose that merely fails to overlap is not: absence claims
    ("nothing in the review flags a regression") and framing sentences are
    lexically unsupportable no matter how honest they are. Weighting both the
    same is how a groundedness detector produces the alert fatigue §9 warns
    about, so they are weighted differently and the reason is here in the code.
    """
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return 1.0, 0.0  # nothing asserted, nothing to contradict

    claim_nums = numbers(claim)
    claim_caps = _named_entities(claim)
    whole = " ".join(context_sentences)

    best = 0.0
    cv = embed_array(claim)
    for premise in (*context_sentences, whole):
        ctx_tokens = set(content_tokens(premise))
        if not ctx_tokens:
            continue
        coverage = sum(1 for t in claim_tokens if t in ctx_tokens) / len(claim_tokens)
        sim = max(0.0, cosine(cv, embed_array(premise)))
        best = max(best, 0.65 * coverage + 0.35 * sim)

    hard_miss = False

    if claim_nums:
        ctx_nums: set[str] = numbers(whole)
        num_support = sum(1 for n in claim_nums if n in ctx_nums) / len(claim_nums)
        if num_support < 1.0:
            hard_miss = True
        # A fabricated figure caps support hard, however well the prose matches.
        best = min(best, 0.15 + 0.85 * num_support)

    if claim_caps:
        ent_support = sum(1 for c in claim_caps if c in whole) / len(claim_caps)
        if ent_support < 1.0:
            hard_miss = True
        best = min(best, 0.45 + 0.55 * ent_support)

    confidence = 1.0 if hard_miss else 0.45
    return max(0.0, min(1.0, best)), confidence


def _nli_support(claim: str, context_sentences: list[str]) -> tuple[float, float]:
    model = _load_nli()
    pairs = [(c, claim) for c in (*context_sentences, " ".join(context_sentences))]
    if not pairs:
        return 0.0, 1.0
    import numpy as np

    logits = model.predict(pairs)
    arr = np.atleast_2d(np.asarray(logits))
    # nli-deberta label order: contradiction, entailment, neutral
    exp = np.exp(arr - arr.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    entail = probs[:, 1]
    contra = probs[:, 0]
    support = float(max(0.0, min(1.0, (entail - contra).max() * 0.5 + 0.5)))
    # An explicit contradiction from the model is hard evidence; mere lack of
    # entailment is not. Same asymmetry as the lexical path.
    confidence = 1.0 if float(contra.max()) > 0.5 else 0.45
    return support, confidence


class GroundednessDetector(DetectorBase):
    name = "groundedness"
    category = RiskCategory.HALLUCINATION
    cost_class = "cheap"
    side = "output"

    # Claims below this support level count as unsupported for the aggregate.
    claim_support_floor = 0.6

    async def _check(self, text: str, ctx: GuardContext) -> Verdict:
        if not ctx.retrieval_context:
            # §9. Unverifiable is not the same as false.
            return self._verdict(
                0.0, evidence="no_grounding_source", abstained=True
            )

        claims = [c for c in (split_sentences(text) or [text]) if _is_checkable(c)]
        if not claims:
            # Nothing in this chunk asserts anything about the source — pure
            # conversational or process text. There is no claim to ground.
            return self._verdict(0.0, evidence="no_checkable_claims")
        context_sentences: list[str] = []
        for doc in ctx.retrieval_context:
            context_sentences.extend(split_sentences(doc) or [doc])
        if not context_sentences:
            return self._verdict(0.0, evidence="empty_context", abstained=True)

        scorer = _nli_support if settings.NLI_BACKEND == "nli" else _lexical_support

        worst_score = 0.0
        unsupported: list[str] = []
        for i, claim in enumerate(claims):
            support, confidence = scorer(claim, context_sentences)
            # Hedged claims assert less, so they are held to a lower bar.
            support = min(1.0, support + 0.25 * hedging_ratio(claim))
            claim_score = (1.0 - support) * confidence
            if support < self.claim_support_floor:
                unsupported.append(f"claim{i}=s{support:.2f}c{confidence:.2f}")
            worst_score = max(worst_score, claim_score)

        if not unsupported and worst_score < 0.05:
            return self._verdict(0.0, evidence="all_claims_supported")

        return self._verdict(
            worst_score,
            evidence=(
                f"unsupported={len(unsupported)}/{len(claims)} "
                + " ".join(unsupported[:4])
            ).strip(),
        )
