"""Policy packs: behaviour varies by use case and geography through
configuration, not code (§7.6).

Hot reload is mtime-based and checked on access. That is the right amount of
machinery here — a filesystem watcher thread would be a second moving part for
no benefit at this scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .. import settings
from ..contracts import Band, BAND_ORDER, RiskCategory


class BandConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    holdback_tokens: int = 0
    sync: bool = False
    guard_budget_ms: int = 200


class EscalationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    budget: float = 0.05
    queue: str = "default_review"


class BiasConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    protected_attributes: list[str] = Field(default_factory=list)
    decision_context: bool = False


class GeographyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    pii_entities: list[str] = Field(default_factory=list)
    require_ledger: bool = False


class ConversationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    ratchet_half_life_turns: float = 4.0
    tool_write_forces_high: bool = True


class RoutingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    cheap_model: str = "replay-cheap"
    frontier_model: str = "replay-frontier"
    complexity_threshold: float = 0.55
    cache_similarity: float = 0.90


class RegenerateConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_attempts: int = 1
    min_headroom_ms: int = 800


class PolicyPack(BaseModel):
    """One use case's governance configuration."""

    model_config = ConfigDict(frozen=True)

    use_case: str
    description: str = ""
    latency_budget_ms: int = 3000
    min_band: Band = Band.LOW
    uncertainty_margin: float = 0.05  # INV-2 rounding zone

    bands: dict[Band, BandConfig] = Field(default_factory=dict)
    detectors: dict[str, list[str]] = Field(default_factory=dict)

    # Band selection: max category score mapped through these boundaries.
    band_thresholds: dict[Band, float] = Field(
        default_factory=lambda: {Band.MEDIUM: 0.35, Band.HIGH: 0.65}
    )
    # Action thresholds per risk category. Overridden by calibration output.
    thresholds: dict[RiskCategory, float] = Field(default_factory=dict)

    masking_enabled: bool = True
    require_masking: bool = True  # if masking is off and PII is outbound -> BLOCK

    escalation: EscalationConfig = EscalationConfig()
    bias: BiasConfig = BiasConfig()
    geography: dict[str, GeographyConfig] = Field(default_factory=dict)
    conversation: ConversationConfig = ConversationConfig()
    routing: RoutingConfig = RoutingConfig()
    regenerate: RegenerateConfig = RegenerateConfig()

    regulated_topics: list[str] = Field(default_factory=list)
    risky_prompt_examples: list[str] = Field(default_factory=list)
    regulated: bool = False

    source_path: str | None = None
    loaded_mtime: float = 0.0

    # -- helpers ---------------------------------------------------------
    def band_config(self, band: Band) -> BandConfig:
        if band in self.bands:
            return self.bands[band]
        # Fall back to the nearest configured band at or above the request, so a
        # pack that omits a band never accidentally relaxes enforcement.
        for b in sorted(self.bands, key=lambda x: BAND_ORDER[x]):
            if BAND_ORDER[b] >= BAND_ORDER[band]:
                return self.bands[b]
        return BandConfig()

    def detectors_for(self, band: Band) -> list[str]:
        names = list(self.detectors.get("floor", []))
        order = [Band.LOW, Band.MEDIUM, Band.HIGH]
        for b in order[: BAND_ORDER[band] + 1]:
            for n in self.detectors.get(b.value, []):
                if n not in names:
                    names.append(n)
        return names

    def threshold(self, category: RiskCategory) -> float:
        return self.thresholds.get(category, DEFAULT_THRESHOLDS[category])

    def pii_entities(self, geography: str) -> list[str]:
        g = self.geography.get(geography)
        return list(g.pii_entities) if g else []

    def requires_ledger(self, geography: str) -> bool:
        g = self.geography.get(geography)
        return bool(g and g.require_ledger)

    def effective_min_band(self) -> Band:
        return self.min_band


DEFAULT_THRESHOLDS: dict[RiskCategory, float] = {
    RiskCategory.PRIVACY: 0.50,
    RiskCategory.INJECTION: 0.60,
    RiskCategory.HALLUCINATION: 0.55,
    RiskCategory.TOXICITY: 0.55,
    RiskCategory.BIAS: 0.60,
}


def _coerce(raw: dict[str, Any], path: Path) -> PolicyPack:
    data = dict(raw)
    if "bands" in data:
        data["bands"] = {Band(k): BandConfig(**v) for k, v in data["bands"].items()}
    if "band_thresholds" in data:
        data["band_thresholds"] = {
            Band(k): float(v) for k, v in data["band_thresholds"].items()
        }
    if "thresholds" in data:
        data["thresholds"] = {
            RiskCategory(k): float(v) for k, v in data["thresholds"].items()
        }
    if "geography" in data:
        data["geography"] = {
            k: GeographyConfig(**v) for k, v in data["geography"].items()
        }
    data["source_path"] = str(path)
    data["loaded_mtime"] = path.stat().st_mtime if path.exists() else 0.0
    return PolicyPack(**data)


class PolicyLoader:
    """Loads packs from a directory, hot-reloading on mtime change."""

    def __init__(self, packs_dir: Path | None = None) -> None:
        self.packs_dir = Path(packs_dir or settings.PACKS_DIR)
        self._cache: dict[str, PolicyPack] = {}

    def _path_for(self, use_case: str) -> Path:
        return self.packs_dir / f"{use_case}.yaml"

    def available(self) -> list[str]:
        return sorted(p.stem for p in self.packs_dir.glob("*.yaml"))

    def load(self, use_case: str) -> PolicyPack:
        path = self._path_for(use_case)
        if not path.exists():
            raise FileNotFoundError(f"no policy pack for use case {use_case!r} at {path}")
        mtime = path.stat().st_mtime
        cached = self._cache.get(use_case)
        if cached is not None and cached.loaded_mtime == mtime:
            return cached
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw.setdefault("use_case", use_case)
        pack = _coerce(raw, path)
        self._cache[use_case] = pack
        return pack

    def load_all(self) -> dict[str, PolicyPack]:
        return {u: self.load(u) for u in self.available()}

    def invalidate(self) -> None:
        self._cache.clear()

    def apply_calibration(self, thresholds: dict[str, dict[str, float]]) -> None:
        """Overlay fitted thresholds onto loaded packs (calibration/fit.py)."""
        for use_case, per_cat in thresholds.items():
            try:
                pack = self.load(use_case)
            except FileNotFoundError:
                continue
            merged = dict(pack.thresholds)
            for cat, val in per_cat.items():
                try:
                    merged[RiskCategory(cat)] = float(val)
                except ValueError:
                    continue
            self._cache[use_case] = pack.model_copy(update={"thresholds": merged})


_loader: PolicyLoader | None = None


def get_loader() -> PolicyLoader:
    global _loader
    if _loader is None:
        _loader = PolicyLoader()
    return _loader
