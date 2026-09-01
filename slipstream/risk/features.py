"""Capability-aware signal extraction (§7.4).

The correction that matters here: **the system must not depend on logprobs.**
Enterprises consume models through an API and most deployments do not return
token logprobs. So there are two signal sets, chosen by a capability probe, and
whichever one ran is recorded on the DecisionRecord so the eval can report the
two modes separately.

Everything in this module is regex, set arithmetic and one cached embedding
lookup. No LLM call — that would blow the 15 ms budget and defeat the design.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum

from ..contracts import ProviderCapabilities, SignalMode
from ..embed import NearestNeighbours
from ..guards import injection as injection_guard
from ..guards import pii as pii_guard
from ..textutil import entity_density, hedging_ratio, overlap_ratio, tokens


class ToolScope(str, Enum):
    """Declared side-effect scope of any tool the request may invoke."""

    NONE = "none"
    READ_ONLY = "read_only"
    WRITE = "write"
    IRREVERSIBLE = "irreversible"


TOOL_SCOPE_RISK: dict[ToolScope, float] = {
    ToolScope.NONE: 0.0,
    ToolScope.READ_ONLY: 0.05,
    ToolScope.WRITE: 0.55,
    ToolScope.IRREVERSIBLE: 0.85,
}


@dataclass
class PromptFeatures:
    topic_hits: list[str] = field(default_factory=list)
    topic_score: float = 0.0
    risky_similarity: float = 0.0
    pii_score: float = 0.0
    pii_types: list[str] = field(default_factory=list)
    injection_score: float = 0.0
    injection_evidence: list[str] = field(default_factory=list)
    toxicity_prior: float = 0.0
    tool_scope: ToolScope = ToolScope.NONE
    tool_risk: float = 0.0
    verifiable: bool = False
    context_chars: int = 0
    conversation_risk: float = 0.0
    prompt_tokens: int = 0
    complexity: float = 0.0
    elapsed_ms: float = 0.0


_ASK_MARKERS = re.compile(
    r"\b(?:compare|analy[sz]e|evaluate|justify|derive|reconcile|forecast|"
    r"multi-?step|step by step|trade-?offs?|why|explain how|synthesi[sz]e)\b",
    re.I,
)

# One index per pack, built on first use and reused. Corpora are ~5 items.
_risky_indexes: dict[str, NearestNeighbours] = {}


def _risky_index(use_case: str, examples: list[str]) -> NearestNeighbours:
    idx = _risky_indexes.get(use_case)
    if idx is None or len(idx.corpus) != len(examples):
        idx = NearestNeighbours(examples)
        _risky_indexes[use_case] = idx
    return idx


def _topic_hits(prompt: str, topics: list[str]) -> list[str]:
    low = prompt.lower()
    return [t for t in topics if t.lower() in low]


def complexity_score(prompt: str) -> float:
    """Cheap routing signal, reused from the same feature pass (§7.10)."""
    toks = tokens(prompt)
    if not toks:
        return 0.0
    length = min(1.0, len(toks) / 120.0)
    markers = min(1.0, len(_ASK_MARKERS.findall(prompt)) / 3.0)
    clauses = min(1.0, prompt.count(",") / 6.0)
    questions = min(1.0, prompt.count("?") / 3.0)
    return min(1.0, 0.35 * length + 0.35 * markers + 0.15 * clauses + 0.15 * questions)


def extract_features(
    prompt: str,
    *,
    use_case: str,
    regulated_topics: list[str],
    risky_prompt_examples: list[str],
    retrieval_context: list[str] | None = None,
    tool_scope: ToolScope = ToolScope.NONE,
    conversation_risk: float = 0.0,
) -> PromptFeatures:
    import time

    t0 = time.perf_counter()
    ctx = retrieval_context or []

    hits = _topic_hits(prompt, regulated_topics)
    topic_score = min(1.0, 0.45 + 0.18 * (len(hits) - 1)) if hits else 0.0

    sim = _risky_index(use_case, risky_prompt_examples).top_k_mean(prompt, k=2)

    pii_hits = pii_guard.scan(prompt)
    pii_score = max((h.confidence for h in pii_hits), default=0.0)

    inj_score, inj_ev = injection_guard.score_text(prompt)

    from ..guards.toxicity import lexicon_score

    tox, _ = lexicon_score(prompt)

    f = PromptFeatures(
        topic_hits=hits,
        topic_score=topic_score,
        risky_similarity=max(0.0, sim),
        pii_score=pii_score,
        pii_types=sorted({h.entity_type for h in pii_hits}),
        injection_score=inj_score,
        injection_evidence=inj_ev,
        toxicity_prior=tox,
        tool_scope=tool_scope,
        tool_risk=TOOL_SCOPE_RISK[tool_scope],
        verifiable=bool(ctx),
        context_chars=sum(len(c) for c in ctx),
        conversation_risk=conversation_risk,
        prompt_tokens=len(tokens(prompt)),
        complexity=complexity_score(prompt),
    )
    f.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return f


# ---------------------------------------------------------------------------
# Hallucination prior — the capability-dependent part
# ---------------------------------------------------------------------------
def _entropy_dispersion(logprobs: list[float]) -> float:
    """Dispersion of token surprisal. Flat-and-high == the model is guessing."""
    vals = [lp for lp in logprobs if lp is not None and math.isfinite(lp)]
    if len(vals) < 4:
        return 0.0
    surprisal = [-lp for lp in vals]
    mean = sum(surprisal) / len(surprisal)
    var = sum((s - mean) ** 2 for s in surprisal) / len(surprisal)
    sd = math.sqrt(var)
    # Normalise into 0..1 against a ~3 nats scale; both a high mean and a high
    # spread push the prior up.
    return max(0.0, min(1.0, 0.6 * (mean / 3.0) + 0.4 * (sd / 2.0)))


def _proxy_prior(text: str, retrieval_context: list[str]) -> float:
    """Signals available to any API-only consumer (§7.4 fallback path)."""
    density = entity_density(text)
    hedge = hedging_ratio(text)
    if retrieval_context:
        overlap = overlap_ratio(text, " ".join(retrieval_context))
        grounding_gap = 1.0 - overlap
    else:
        # Nothing to check against. The prior leans on specificity alone: a very
        # concrete claim with no source is where fabrication hides.
        grounding_gap = 0.45

    from ..guards.consistency import self_contradiction

    contra, _ = self_contradiction(text)

    prior = 0.45 * grounding_gap + 0.25 * density + 0.20 * contra + 0.10 * hedge
    return max(0.0, min(1.0, prior))


def hallucination_prior(
    text: str,
    *,
    capabilities: ProviderCapabilities,
    logprobs: list[float] | None = None,
    retrieval_context: list[str] | None = None,
) -> tuple[float, SignalMode]:
    if capabilities.supports_logprobs and logprobs:
        return _entropy_dispersion(logprobs), SignalMode.LOGPROBS
    return _proxy_prior(text, retrieval_context or []), SignalMode.PROXY
