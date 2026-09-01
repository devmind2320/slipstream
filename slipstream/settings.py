"""Process-wide settings and paths.

Every heavyweight model is opt-in behind an env flag. The default path is pure
Python and runs offline on a clean clone with no downloads (INV-5). See
ASSUMPTIONS.md for why this deviates from §4 of the spec.
"""

from __future__ import annotations

import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent

PACKS_DIR = Path(os.environ.get("SLIPSTREAM_PACKS_DIR", PKG_ROOT / "policy" / "packs"))
DATA_DIR = Path(os.environ.get("SLIPSTREAM_DATA_DIR", REPO_ROOT / "data"))
FIXTURES_DIR = Path(
    os.environ.get("SLIPSTREAM_FIXTURES_DIR", PKG_ROOT / "eval" / "datasets" / "fixtures")
)
RESULTS_DIR = Path(os.environ.get("SLIPSTREAM_RESULTS_DIR", PKG_ROOT / "eval" / "results"))
CALIBRATION_DIR = Path(
    os.environ.get("SLIPSTREAM_CALIBRATION_DIR", PKG_ROOT / "calibration")
)

LEDGER_PATH = Path(os.environ.get("SLIPSTREAM_LEDGER", DATA_DIR / "ledger.sqlite3"))
THRESHOLDS_PATH = CALIBRATION_DIR / "thresholds.json"
CALIBRATION_SET_PATH = DATA_DIR / "calibration_set.jsonl"

# Salt for entity hashing in the ledger. Overridable so a deployment can rotate
# it; fixed default keeps eval runs reproducible (INV-5 / determinism test).
ENTITY_SALT = os.environ.get("SLIPSTREAM_ENTITY_SALT", "slipstream-dev-salt")


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- optional model backends -------------------------------------------------
# "hash"  -> deterministic hashed n-gram vectors, offline, ~0.05 ms  (DEFAULT)
# "st"    -> sentence-transformers/all-MiniLM-L6-v2 (downloads on first run)
EMBED_BACKEND = os.environ.get("SLIPSTREAM_EMBED_BACKEND", "hash").strip().lower()

# "lexical" -> lexical-entailment proxy, offline (DEFAULT)
# "nli"     -> cross-encoder/nli-deberta-v3-small (downloads on first run)
NLI_BACKEND = os.environ.get("SLIPSTREAM_NLI_BACKEND", "lexical").strip().lower()

# "lexicon" -> curated lexicon, offline (DEFAULT)
# "bert"    -> unitary/toxic-bert (downloads on first run)
TOXICITY_BACKEND = os.environ.get("SLIPSTREAM_TOXICITY_BACKEND", "lexicon").strip().lower()

# --- detector latency model --------------------------------------------------
# The offline default engines (regex PII, lexical-entailment NLI, lexicon
# toxicity) run in ~1 ms, one to two orders of magnitude faster than the
# transformer models they stand in for. Measuring the scheduler against 1 ms
# detectors would make all three evaluation arms look identical and would prove
# nothing: the whole question is what happens when verification is *expensive*.
#
# So by default each detector's wall-clock is floored at the §4 CPU target for
# the model it substitutes for. The DECISIONS are always the real engine's; only
# the timing is modelled, and results.json records which profile ran.
#
#   "spec"     (default) floor latency at the targets below
#   "measured"           report raw measured latency, no floor
LATENCY_PROFILE = os.environ.get("SLIPSTREAM_LATENCY_PROFILE", "spec").strip().lower()

# cost_class -> modelled floor in ms.
#   floor      presidio-analyzer + spaCy en_core_web_sm over a chunk
#   cheap      cross-encoder/nli-deberta-v3-small, §4 target "< 80 ms per chunk"
#   expensive  multi-sample agreement over 3 completions
LATENCY_TARGETS_MS: dict[str, float] = {
    "floor": float(os.environ.get("SLIPSTREAM_LAT_FLOOR_MS", "6")),
    "cheap": float(os.environ.get("SLIPSTREAM_LAT_CHEAP_MS", "55")),
    "expensive": float(os.environ.get("SLIPSTREAM_LAT_EXPENSIVE_MS", "200")),
}

USE_PRESIDIO = _flag("SLIPSTREAM_USE_PRESIDIO", False)
USE_REDIS = _flag("SLIPSTREAM_USE_REDIS", False)
OTEL_CONSOLE = _flag("SLIPSTREAM_OTEL_CONSOLE", False)
OTEL_FILE = Path(os.environ.get("SLIPSTREAM_OTEL_FILE", DATA_DIR / "spans.jsonl"))

PROVIDER = os.environ.get("SLIPSTREAM_PROVIDER", "replay").strip().lower()
OPENAI_BASE_URL = os.environ.get("SLIPSTREAM_OPENAI_BASE_URL", "")
OPENAI_API_KEY = os.environ.get("SLIPSTREAM_OPENAI_API_KEY", "")

DEFAULT_SEED = int(os.environ.get("SLIPSTREAM_SEED", "1729"))


def ensure_dirs() -> None:
    for d in (DATA_DIR, RESULTS_DIR, FIXTURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def backend_summary() -> dict[str, str]:
    """Reported in the eval output so results are never ambiguous about what ran."""
    return {
        "embed": EMBED_BACKEND,
        "nli": NLI_BACKEND,
        "toxicity": TOXICITY_BACKEND,
        "pii": "presidio+regex" if USE_PRESIDIO else "regex+gazetteer",
        "provider": PROVIDER,
        "latency_profile": LATENCY_PROFILE,
        "latency_targets_ms": ",".join(
            f"{k}={v:g}" for k, v in sorted(LATENCY_TARGETS_MS.items())
        ),
    }
