"""Slipstream data contracts.

Everything else in the system imports from here. Defined first, per spec §6/§15.

Design notes that are load-bearing:

* ``RiskVector.scores`` is multi-label by construction. A fabricated detail about
  a real person scores on BOTH ``HALLUCINATION`` and ``PRIVACY``. Do not collapse
  it to a single argmax label anywhere downstream.
* ``Verdict.abstained`` is not a low score. "I cannot check this" and "this is
  wrong" are different states and route to different actions (§9).
* ``DecisionRecord`` never carries raw PII. Entity *types*, offsets and salted
  hashes only.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "RiskCategory",
    "RiskVector",
    "Band",
    "StreamPolicy",
    "Verdict",
    "Action",
    "EntityHit",
    "ChunkTiming",
    "SignalMode",
    "DecisionRecord",
    "GuardContext",
    "ProviderCapabilities",
    "canonical_json",
    "ACTION_SEVERITY",
    "max_severity",
    "BAND_ORDER",
    "max_band",
]


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------
class RiskCategory(str, Enum):
    HALLUCINATION = "hallucination"
    PRIVACY = "privacy"
    BIAS = "bias"
    TOXICITY = "toxicity"
    INJECTION = "injection"


class RiskVector(BaseModel):
    """Multi-label. Categories overlap by design; a fabricated detail about a
    real person scores on BOTH hallucination and privacy. DO NOT collapse to
    a single label."""

    model_config = ConfigDict(frozen=True)

    scores: dict[RiskCategory, float] = Field(default_factory=dict)
    verifiable: bool = False
    confidence: float = 0.5

    @field_validator("scores")
    @classmethod
    def _bounded(cls, v: dict[RiskCategory, float]) -> dict[RiskCategory, float]:
        for k, s in v.items():
            if not 0.0 <= s <= 1.0:
                raise ValueError(f"score for {k} out of range: {s}")
        return v

    def max_score(self) -> float:
        return max(self.scores.values(), default=0.0)

    def top_category(self) -> RiskCategory | None:
        if not self.scores:
            return None
        return max(self.scores.items(), key=lambda kv: (kv[1], kv[0].value))[0]

    def above(self, threshold: float) -> set[RiskCategory]:
        return {c for c, s in self.scores.items() if s >= threshold}


class Band(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


BAND_ORDER: dict[Band, int] = {Band.LOW: 0, Band.MEDIUM: 1, Band.HIGH: 2}


def max_band(*bands: Band) -> Band:
    """INV-2 helper: uncertainty always resolves to the *higher* band."""
    return max(bands, key=lambda b: BAND_ORDER[b])


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------
class StreamPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    band: Band
    holdback_tokens: int = 0  # 0 == stream_first (release, scan behind)
    detectors: list[str] = Field(default_factory=list)
    sync: bool = False  # block release until verdict
    masking_enabled: bool = True
    escalation_armed: bool = False
    guard_budget_ms: int = 200


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
class EntityHit(BaseModel):
    """A PII/entity match. Carries type + offsets + salted hash, NEVER the value."""

    model_config = ConfigDict(frozen=True)

    entity_type: str
    start: int
    end: int
    value_hash: str
    confidence: float = 1.0


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    detector: str
    category: RiskCategory
    score: float
    latency_ms: float
    evidence: str | None = None  # span offsets / entity type, never the raw secret
    abstained: bool = False  # detector could not decide (§9)
    entities: list[EntityHit] = Field(default_factory=list)

    def triggered(self, threshold: float) -> bool:
        return (not self.abstained) and self.score >= threshold


class Action(str, Enum):
    ALLOW = "allow"
    ANNOTATE = "annotate"
    MASK = "mask"
    REGENERATE = "regenerate"
    ESCALATE = "escalate"
    BLOCK = "block"


# Action selection takes the *max severity* across triggered categories (§9),
# never the first match.
ACTION_SEVERITY: dict[Action, int] = {
    Action.ALLOW: 0,
    Action.ANNOTATE: 1,
    Action.MASK: 2,
    Action.REGENERATE: 3,
    Action.ESCALATE: 4,
    Action.BLOCK: 5,
}


def max_severity(actions: list[Action]) -> Action:
    if not actions:
        return Action.ALLOW
    return max(actions, key=lambda a: (ACTION_SEVERITY[a], a.value))


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------
class ChunkTiming(BaseModel):
    """Per-chunk scheduler instrumentation. Required for the §10 headline metric."""

    model_config = ConfigDict(frozen=True)

    index: int
    tokens: int
    gen_ms: float  # wall time generating the *next* chunk
    guard_ms: float  # wall time running detectors on *this* chunk
    guard_finished_first: bool  # guard(N) done before chunk N+1 was cut
    # Did this chunk actually have a race to win? A synchronous policy blocks
    # generation until the verdict lands, so there is no concurrent window and
    # "the guard lost" is meaningless. Unraced chunks are excluded from the
    # headline overlap metric and from backpressure accounting.
    raced: bool = True
    released: bool
    leaked: bool = False  # emitted before its verdict was known, then found dirty
    degraded: bool = False


class SignalMode(str, Enum):
    """Which hallucination-prior signal set was available (§7.4)."""

    LOGPROBS = "logprobs"
    PROXY = "proxy"


class ProviderCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    supports_logprobs: bool = False
    supports_multi_sample: bool = False
    supports_system_prompt: bool = True
    name: str = "unknown"


# --------------------------------------------------------------------------
# Guard context
# --------------------------------------------------------------------------
class GuardContext(BaseModel):
    """Everything a detector may look at. Input/output layer only (§9)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    prompt: str = ""
    retrieval_context: list[str] = Field(default_factory=list)
    use_case: str = "internal_knowledge"
    geography: str = "IN"
    band: Band = Band.MEDIUM
    pii_entities: list[str] = Field(default_factory=list)
    protected_attributes: list[str] = Field(default_factory=list)
    decision_context: bool = False
    capabilities: ProviderCapabilities = ProviderCapabilities()
    budget_ms: int = 200
    is_prompt_side: bool = False

    @property
    def verifiable(self) -> bool:
        return bool(self.retrieval_context)


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------
def canonical_json(obj: Any) -> str:
    """Stable serialisation for hashing. Sorted keys, no whitespace drift."""

    def default(o: Any) -> Any:
        if isinstance(o, BaseModel):
            return o.model_dump(mode="json")
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, set):
            return sorted(o)
        raise TypeError(f"not serialisable: {type(o)!r}")

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=default)


class DecisionRecord(BaseModel):
    record_id: str
    prev_hash: str = ""
    record_hash: str = ""

    trace_id: str = ""
    use_case: str = "internal_knowledge"
    geography: str = "IN"
    conversation_id: str | None = None
    turn_index: int = 0

    risk_vector: RiskVector
    policy: StreamPolicy
    verdicts: list[Verdict] = Field(default_factory=list)
    action: Action = Action.ALLOW

    entities_masked: list[str] = Field(default_factory=list)  # TYPES, never values
    degraded_mode: bool = False
    abstained: bool = False
    regenerated: bool = False
    escalated: bool = False
    signal_mode: SignalMode = SignalMode.PROXY

    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    guard_overhead_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    model_used: str = "replay"
    arm: str = "adaptive"

    chunk_timings: list[ChunkTiming] = Field(default_factory=list)
    cache_hit: bool = False
    cache_gated_miss: bool = False
    leaked_chunks: int = 0

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def payload_for_hash(self) -> str:
        """Canonical form excluding the hash fields themselves."""
        d = self.model_dump(mode="json", exclude={"record_hash"})
        return canonical_json(d)

    def compute_hash(self) -> str:
        return hashlib.sha256(
            (self.prev_hash + self.payload_for_hash()).encode("utf-8")
        ).hexdigest()


StreamEventType = Literal["chunk", "record", "annotation", "blocked", "done"]
