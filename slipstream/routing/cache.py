"""Entity-gated semantic cache (§7.10).

The failure mode this exists to prevent: two prompts sit at cosine 0.97 and
differ only in the account number, the date, or the name. A plain semantic cache
serves the wrong customer's answer with high confidence. That is a privacy
incident produced by a performance optimisation.

So the gate is: embedding similarity proposes, entity agreement disposes. If the
identifiers, figures or dates in the query differ from the ones in the cached
prompt, it is a miss regardless of how close the vectors are. Those **gated
misses are counted**, because that counter is the argument for the design.

Redis is behind a flag and off by default (§4); the in-process dict is the
default and is what the eval measures.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .. import settings
from ..embed import cosine, embed_array
from ..guards.pii import scan
from ..textutil import numbers


@dataclass
class CacheStats:
    lookups: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    gated_misses: int = 0  # similar enough, but the entities disagreed
    misses: int = 0

    @property
    def hits(self) -> int:
        return self.exact_hits + self.semantic_hits

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "lookups": self.lookups,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "gated_misses": self.gated_misses,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass
class _Entry:
    prompt: str
    value: Any
    entities: frozenset[str]
    vector: Any
    created_at: float = field(default_factory=time.monotonic)


def entity_signature(text: str) -> frozenset[str]:
    """The identifiers that must agree for a cache hit to be safe.

    Salted hashes of detected PII entities plus every numeric literal. Names and
    figures are exactly what makes two near-identical prompts different requests.
    """
    sig = {f"{h.entity_type}:{h.value_hash}" for h in scan(text)}
    sig |= {f"NUM:{n}" for n in numbers(text)}
    return frozenset(sig)


class SemanticCache:
    def __init__(
        self,
        *,
        similarity_threshold: float = 0.90,
        max_entries: int = 2048,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        self.stats = CacheStats()
        self._redis = self._connect_redis() if settings.USE_REDIS else None

    def _connect_redis(self):
        try:
            import redis  # optional

            return redis.Redis.from_url(
                settings.__dict__.get("REDIS_URL", "redis://localhost:6379/0")
            )
        except Exception:  # noqa: BLE001 - flag is off by default; degrade quietly
            return None

    def _key(self, prompt: str, use_case: str) -> str:
        return f"{use_case}::{prompt.strip().lower()}"

    def get(self, prompt: str, use_case: str) -> tuple[Any | None, str]:
        """Returns (value, outcome) where outcome is one of
        exact | semantic | gated_miss | miss."""
        self.stats.lookups += 1
        key = self._key(prompt, use_case)

        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            self.stats.exact_hits += 1
            return entry.value, "exact"

        q_entities = entity_signature(prompt)
        q_vec = embed_array(prompt)
        best: _Entry | None = None
        best_sim = 0.0
        gated = False

        for k, e in self._entries.items():
            if not k.startswith(f"{use_case}::"):
                continue
            sim = cosine(q_vec, e.vector)
            if sim < self.similarity_threshold:
                continue
            if e.entities != q_entities:
                # Close enough to be dangerous. This is the whole point.
                gated = True
                continue
            if sim > best_sim:
                best, best_sim = e, sim

        if best is not None:
            self.stats.semantic_hits += 1
            return best.value, "semantic"
        if gated:
            self.stats.gated_misses += 1
            return None, "gated_miss"
        self.stats.misses += 1
        return None, "miss"

    def put(self, prompt: str, use_case: str, value: Any) -> None:
        key = self._key(prompt, use_case)
        self._entries[key] = _Entry(
            prompt=prompt,
            value=value,
            entities=entity_signature(prompt),
            vector=embed_array(prompt),
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._entries)
