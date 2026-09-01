PY ?= python
SEED ?= 1729
SCALE ?= 1.0
ALPHA ?= 0.10
CONCURRENCY ?= 24
PORT ?= 8000

.DEFAULT_GOAL := help
.PHONY: help install demo demo-injection demo-benign serve data eval eval-fast \
        eval-deterministic fit recalibrate test test-fast verify info clean report

help:
	@echo "Slipstream — adaptive verification scheduling for streaming LLM responses"
	@echo ""
	@echo "  make install     install pinned dependencies (no model downloads)"
	@echo "  make demo        offline A/B demo — no network, no API key"
	@echo "  make serve       gateway + dashboard on http://127.0.0.1:$(PORT)"
	@echo "  make eval        regenerate the full three-arm results table"
	@echo "  make eval-fast   the same at 25% corpus scale, for a quick look"
	@echo "  make test        full test suite"
	@echo "  make verify      walk the ledger hash chain"
	@echo ""
	@echo "  Everything runs offline. Override SEED=, SCALE=, ALPHA= as needed."

install:
	$(PY) -m pip install -r requirements.txt

# ---------------------------------------------------------------------------
# INV-5: both of these run end to end with no network access and no API keys.
# ---------------------------------------------------------------------------
demo:
	$(PY) -m slipstream demo --scenario compensation

demo-injection:
	$(PY) -m slipstream demo --scenario injection

demo-benign:
	$(PY) -m slipstream demo --scenario benign

serve:
	@echo "dashboard -> http://127.0.0.1:$(PORT)/"
	$(PY) -m slipstream serve --port $(PORT)

data:
	$(PY) -m slipstream data --seed $(SEED) --scale $(SCALE)

# Regenerates the corpus, refits the conformal thresholds and rebuilds
# eval/results/{results.json,report.md} from scratch.
eval:
	$(PY) -m slipstream eval --seed $(SEED) --scale $(SCALE) --alpha $(ALPHA) \
		--concurrency $(CONCURRENCY)

eval-fast:
	$(PY) -m slipstream eval --seed $(SEED) --scale 0.25 --alpha $(ALPHA) \
		--concurrency $(CONCURRENCY)

# Disables backpressure degradation and guard-budget timeouts — the two
# wall-clock-dependent inputs to a decision — so the run is bit-for-bit
# reproducible. Costs the scheduler's ability to shed load; see ASSUMPTIONS.md.
eval-deterministic:
	$(PY) -m slipstream.eval.harness --seed $(SEED) --scale $(SCALE) \
		--alpha $(ALPHA) --concurrency $(CONCURRENCY) --deterministic

report:
	@cat slipstream/eval/results/report.md

fit:
	$(PY) -m slipstream fit --alpha $(ALPHA) --seed $(SEED) --report

recalibrate:
	$(PY) -m slipstream recalibrate --alpha $(ALPHA) --seed $(SEED)

# ---------------------------------------------------------------------------
test:
	$(PY) -m pytest tests/

test-fast:
	$(PY) -m pytest tests/ -x -q -k "not determinism"

verify:
	$(PY) -m slipstream ledger verify

info:
	$(PY) -m slipstream info

clean:
	rm -rf data/ .pytest_cache/ slipstream/eval/datasets/fixtures/generated.jsonl
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
