# Slipstream

Slipstream is a model-agnostic streaming gateway that decides, per request, how
much verification a response deserves — and schedules that verification to run
concurrently with token generation instead of in front of it. A sub-15 ms
prompt-side estimator picks a verification band, the scheduler holds back exactly
as much of the stream as that band requires, and split-conformal calibration
turns the resulting thresholds into a stated risk bound rather than a tuned
guess.

Streaming guardrails already exist. What NVIDIA NeMo Guardrails and AWS Bedrock
Guardrails share is that the safety–latency trade is fixed at configuration time
and applied uniformly to all traffic — one chunk size, one buffer depth, one set
of checks, whether the request is "what are your opening hours" or "summarise
this employee's compensation record". Slipstream makes that setting a calibrated
per-request decision. Against a fixed-synchronous baseline it closes 64% of the
time-to-first-token gap while holding that baseline's PII interception rate and
groundedness recall exactly.

Built for the Accenture Innovation Challenge 2026, Round 2, PS1 ControlPlane.ai.

For the full source, see the
[project repository](https://github.com/YOUR-USERNAME/slipstream).

Submit bug reports and feature suggestions, or track changes, in the
[issue tracker](https://github.com/YOUR-USERNAME/slipstream/issues).


## Table of contents

- Requirements
- Recommended backends
- Installation
- Configuration
- How it works
- Key features
- Results
- Design decisions
- Governance and audit
- Tests
- Limitations
- Repository layout
- Troubleshooting
- FAQ
- Maintainers


## Requirements

Slipstream runs offline on a clean clone. There is no API key, no network
access and no model download on the default path — first-run download size is
0 bytes.

This project requires the following:

- Python 3.11 or newer
- The packages pinned in [requirements.txt](requirements.txt), all pure-Python
  or small wheels: FastAPI, Uvicorn, Starlette, Pydantic, PyYAML, NumPy, httpx,
  the OpenTelemetry API and SDK, and pytest
- GNU Make, optionally — every target has a PowerShell equivalent in
  [make.ps1](make.ps1) for Windows users without `make`

No GPU is required, and no model weights are downloaded.


## Recommended backends

None of these are required. The default engines are deterministic, pure-Python
stand-ins that keep the offline guarantee intact; the packages below replace
them with the heavier models named in the problem statement, at the cost of
roughly 840 MB of first-run downloads. All are documented in
[requirements-optional.txt](requirements-optional.txt) and enabled by
environment variable.

- [sentence-transformers](https://www.sbert.net/): replaces the offline
  embedding backend with `all-MiniLM-L6-v2`, improving the semantic kNN signal
  in the risk estimator and the routing cache.
- [nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small):
  replaces lexical entailment with a real NLI model, which materially improves
  groundedness recall on paraphrased fabrications.
- [unitary/toxic-bert](https://huggingface.co/unitary/toxic-bert): replaces the
  lexicon-based toxicity engine with a trained classifier.
- [Presidio](https://microsoft.github.io/presidio/): replaces the built-in PII
  recogniser set with Presidio plus spaCy, adding broader entity coverage.
- [Redis](https://redis.io/): replaces the in-process routing cache with a
  shared one, which matters only in a multi-worker deployment.


## Installation

Install as you would normally install a Python project.

```bash
git clone https://github.com/YOUR-USERNAME/slipstream.git
cd slipstream
pip install -r requirements.txt
make demo
```

On Windows without `make`:

```powershell
py -3.11 -m pip install -r requirements.txt
.\make.ps1 demo
```

Every Makefile target has a `.\make.ps1 <target>` equivalent. The available
targets are:

| Command | What it does |
|---|---|
| `make demo` | Offline A/B: the same request through `fixed_async` and `adaptive`, side by side |
| `make serve` | Gateway and live dashboard at `http://127.0.0.1:8000/` |
| `make eval` | Regenerates the corpus, refits calibration, rebuilds the results table |
| `make eval-fast` | The same at 25% corpus scale, about 30 seconds |
| `make eval-deterministic` | Bit-for-bit reproducible run with wall-clock paths disabled |
| `make test` | The full suite, 139 tests |
| `make verify` | Walks the ledger hash chain and reports the first break |
| `make info` | Prints active backends, detectors and the registry fingerprint |
| `make recalibrate` | Refits thresholds from captured reviewer feedback |
| `make clean` | Removes generated data, fixtures and caches |

To confirm the installation, run `make info`. It prints the active backends and
the detector-registry fingerprint without touching the network.


## Configuration

Behaviour varies by use case and geography through YAML, not code. The packs
live in [slipstream/policy/packs/](slipstream/policy/packs/) and are hot-reloaded
on change.

To change how a use case behaves:

1. Open the pack for your use case in
   [slipstream/policy/packs/](slipstream/policy/packs/) — one of
   `customer_support.yaml`, `internal_knowledge.yaml` or
   `decision_support.yaml`.
1. Set `min_band` to the lowest verification band this use case may ever run in.
   A regulated use case should never reach `low`.
1. Tune `holdback_tokens`, `sync` and `guard_budget_ms` per band. Deeper
   hold-back gives detectors a larger window to win their race against the next
   chunk, at the cost of time-to-first-token.
1. Declare which detectors run per band. Whatever you list under `floor` runs
   inline in every band and cannot be degraded away.
1. List the PII entity types for each geography you serve.
1. Save the file. The loader picks up the change without a restart.

```yaml
use_case: decision_support
min_band: medium                    # never runs in LOW
uncertainty_margin: 0.08            # INV-2 rounding zone
bands:
  medium: { holdback_tokens: 64,  sync: false, guard_budget_ms: 120 }
  high:   { holdback_tokens: 256, sync: true,  guard_budget_ms: 400 }
detectors:
  floor: [pii, injection]           # always on, all bands
  medium: [groundedness]
  high: [groundedness, consistency, bias]
escalation: { budget: 0.06, queue: decision_review }
geography:
  IN: { pii_entities: [PAN, AADHAAR, PHONE_IN, EMAIL, ACCOUNT] }
  EU: { pii_entities: [EMAIL, PHONE, IBAN, NATIONAL_ID], require_ledger: true }
```

Runtime behaviour is set by environment variable. All optional backends are off
by default:

```bash
SLIPSTREAM_EMBED_BACKEND=st        # sentence-transformers/all-MiniLM-L6-v2
SLIPSTREAM_NLI_BACKEND=nli         # cross-encoder/nli-deberta-v3-small
SLIPSTREAM_TOXICITY_BACKEND=bert   # unitary/toxic-bert
SLIPSTREAM_USE_PRESIDIO=1          # presidio-analyzer + spaCy
SLIPSTREAM_USE_REDIS=1             # shared routing cache
SLIPSTREAM_PROVIDER=openai         # any OpenAI-compatible /v1/chat/completions
SLIPSTREAM_OPENAI_BASE_URL=...     # provider endpoint
SLIPSTREAM_OPENAI_API_KEY=...      # only needed for a live provider
SLIPSTREAM_LATENCY_PROFILE=measured  # disable modelled detector latency
SLIPSTREAM_SEED=1729               # corpus and calibration seed
SLIPSTREAM_LEDGER=...              # ledger database path
SLIPSTREAM_ENTITY_SALT=...         # salt for hashed entity references
SLIPSTREAM_OTEL_CONSOLE=1          # emit spans to stdout
```

The paths `SLIPSTREAM_DATA_DIR`, `SLIPSTREAM_PACKS_DIR`,
`SLIPSTREAM_FIXTURES_DIR`, `SLIPSTREAM_CALIBRATION_DIR` and
`SLIPSTREAM_RESULTS_DIR` relocate generated and loaded artefacts if the
defaults do not suit your layout.


## How it works

```
                                prompt
                                   │
              ┌────────────────────▼────────────────────┐
              │  Tier-0 risk estimator      < 15 ms     │   regex · keyword · embedding kNN
              │  risk/estimator.py                      │   PII-in-prompt · tool scope
              │                                         │   retrieval attached? · conversation
              │  multi-label RiskVector ─► Band         │   ratchet
              │  INV-2: uncertainty rounds UP           │   NO LLM CALL
              └────────────────────┬────────────────────┘
                                   │  Band ─► StreamPolicy (from the YAML pack)
                                   │  holdback · detectors · sync · budget
        ┌──────────────────────────┼──────────────────────────┐
        │                          ▼                          │
        │   ╔═══ FLOOR ═══════════════════════════════════╗   │  INV-1
        │   ║  pii + injection, inline, EVERY band,       ║   │  never skipped,
        │   ║  before a single token is generated         ║   │  never degraded away
        │   ╚═════════════════════════════════════════════╝   │
        │                          │                          │
        │              ┌───────────▼───────────┐              │
        │              │  router · cache       │              │  entity-gated:
        │              │  routing/             │              │  similar ≠ same
        │              └───────────┬───────────┘              │
        │                          ▼                          │
        │   ┌──────────────────────────────────────────────┐  │
        │   │        HOLD-BACK SCHEDULER                   │  │
        │   │        gateway/scheduler.py                  │  │
        │   │                                              │  │
        │   │   provider ──tokens──► buffer ──► chunk N    │  │
        │   │                                     │        │  │
        │   │        ┌────────────────────────────┴─────┐  │  │
        │   │        │ generate chunk N+1  ║  guard(N)  │  │  │  ◄── the point:
        │   │        │   (provider)        ║ (detectors)│  │  │      concurrent,
        │   │        └────────────────────────────┬─────┘  │  │      not sequential
        │   │                                     ▼        │  │
        │   │            clean ─► release   dirty ─► never │  │
        │   │                                     emitted  │  │
        │   │   guards lose k races ─► degrade to floor,   │  │
        │   │   never stall                                │  │
        │   └──────────────────────┬───────────────────────┘  │
        │                          ▼                          │
        │   ┌──────────────────────────────────────────────┐  │
        │   │  action matrix   allow · annotate · mask ·   │  │  max severity across
        │   │  actions/        regenerate · escalate ·     │  │  triggered categories,
        │   │                  block                       │  │  never first match
        │   │  abstain ≠ violation: routes separately      │  │
        │   └──────────────────────┬───────────────────────┘  │
        └──────────────────────────┼──────────────────────────┘
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │  ledger/  append-only, hash-chained, gen_ai.* spans   │
        │  entity TYPES only — never values                     │
        └──────────────────────┬───────────────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────────────┐
        │  calibration/  split-conformal ─► thresholds          │
        │  feedback/     human overrides ─► recalibrate ─► drift│
        └──────────────────────────────────────────────────────┘
```


### The mechanism, concretely

Chunk *N* is held. Its detectors run while the provider generates chunk *N+1*.
If the verdict is clean by the time chunk *N+1* is cut, chunk *N* is released
and verification cost nothing in wall-clock. If it is dirty, chunk *N* is never
emitted.

That race is what buys the latency back, so it is measured rather than asserted:
`pct_chunks_guard_finished_first` is a headline metric, broken out per band — a
shallow LOW hold-back gives detectors a small window, a deep HIGH hold-back gives
them a large one — and a test asserts the wall-clock overlap directly.


### The floor

The obvious attack on adaptive verification is "what happens when your risk
estimator is wrong?" The answer is INV-1: prompt-injection and outbound-PII
detection run inline, at full depth, in every band including the lowest. A
misjudged risk estimate can cost latency. It cannot open the leak path. That is
enforced in the scheduler and asserted in the test suite, not promised in a
README.


### The demo

`make demo` runs an internal finance agent asked to summarise a compensation
record, through two arms side by side.

- Left, `fixed_async`: the response streams immediately. The salary figure, the
  payroll account, the PAN and the contact details all reach the screen. The
  violation is flagged afterwards, and the pane keeps showing what was displayed
  before the correction — erasing a response does not un-display it.
- Right, `adaptive`: the prompt scores HIGH, the buffer deepens to 224 tokens,
  and every one of those entities is masked before the first chunk is released.

Both panes show live time-to-first-token and added-latency counters, and the
decision record for each with its place in the hash chain visible. `make serve`
gives the same thing in a browser over server-sent events.


## Key features

| Piece | Where | Note |
|---|---|---|
| Data contracts | [contracts.py](slipstream/contracts.py) | Written first. Multi-label throughout; abstention is a first-class state |
| Hold-back scheduler | [gateway/scheduler.py](slipstream/gateway/scheduler.py) | The differentiator. Three regimes, one code path |
| Tier-0 estimator | [risk/estimator.py](slipstream/risk/estimator.py) | p95 measured under 15 ms; no LLM in the path, and a test guarding against one being added |
| Conversation ratchet | [risk/conversation.py](slipstream/risk/conversation.py) | Band floor rises across turns and decays on a half-life; write-scoped tools force HIGH |
| Detectors | [guards/](slipstream/guards/) | pii and injection (floor) · groundedness · toxicity · bias (cheap) · consistency (expensive) |
| Policy packs | [policy/packs/](slipstream/policy/packs/) | Three use cases by geography, hot-reloaded from YAML |
| Conformal calibration | [calibration/conformal.py](slipstream/calibration/conformal.py) | With a mandatory empirical coverage check |
| Action matrix | [actions/matrix.py](slipstream/actions/matrix.py) | Max severity across triggered categories |
| Hash-chained ledger | [ledger/store.py](slipstream/ledger/store.py) | Append-only enforced by SQLite triggers, not convention |
| Three-arm harness | [eval/harness.py](slipstream/eval/harness.py) | Score, fit, run, probe, report |
| Corpus generator | [eval/datasets/generate.py](slipstream/eval/datasets/generate.py) | Committed; the corpus itself is not |
| Dashboard | [dashboard/index.html](slipstream/dashboard/index.html) | One file, vanilla JS, server-sent events, no build step |


## Results

The full table is in
[the evaluation report](slipstream/eval/results/report.md), with raw figures in
[results.json](slipstream/eval/results/results.json).

All three arms run the same detectors at the same thresholds against the same
provider. This is INV-3: the harness asserts the detector-registry fingerprint is
identical across arms and records it in the results. Only the scheduling policy
differs.

| Metric | `fixed_sync`<br>safe, slow | `fixed_async`<br>fast, leaky | `adaptive`<br>**Slipstream** |
|---|---|---|---|
| TTFT p50, benign traffic | 495.6 ms | 31.7 ms | **198.6 ms** |
| p50 added latency, benign | 241.5 ms | 219.2 ms | **89.5 ms** |
| p95 added latency, benign | 508.6 ms | 341.7 ms | **240.8 ms** |
| **PII masked before emission** | 100.0% | **0.0%** | **100.0%** |
| PII items that leaked | 0 | **245** | **0** |
| PII interception rate | 100.0% | 100.0% | 100.0% |
| Groundedness recall | 99.1% | 100.0% | 99.1% |
| Multi-label set-exact accuracy | 85.9% | 84.5% | 85.9% |
| Set-exact on the overlapping slice | 100.0% | 100.0% | 100.0% |
| Abstention rate (correct / spurious) | 24.1% (100% / 7.6%) | 21.0% (100% / 3.8%) | 21.1% (100% / 3.9%) |
| Interventions per 100 interactions | 44.5 | 43.2 | 44.5 |
| Chunks where the guard finished first | n/a (synchronous) | 100.0% | 50.4% |
| Degraded-mode incidence | 0.0% | 0.0% | 20.7% |
| Net token cost delta | −58.5% | −58.5% | −58.5% |

Adaptive closes 64% of the time-to-first-token gap between the safe and fast
baselines while holding the safe baseline's interception rate and recall exactly.

Two findings are worth more than that headline:

- `fixed_async` intercepts 100% of outbound PII and prevents 0% of it. It detects
  every leak — after the tokens are on the screen. Interception and prevention
  are different metrics, and the gap between them is the entire argument for
  hold-back scheduling.
- Under `--deterministic`, `adaptive` and `fixed_sync` produce the same decision
  digest: bit-identical decisions on every item. Adaptive is not approximating
  the safe baseline on this corpus; it is reproducing it, at 40% of its
  time-to-first-token.


### Reproducing the results

```bash
make eval          # about 2 minutes, offline, regenerates corpus, calibration and both files
```

Seed `1729`, corpus 2,290 items with 1,145 held-out test items per arm. Backends
and platform are recorded in `results.json` under `run`, and the exact command is
printed in the report as `run.reproduce`.

Decisions are reproducible; wall-clock is not. Two mechanisms make a decision
depend on machine load — backpressure degradation and guard-budget timeouts —
and both are real enforcement the specification requires. The results carry a
`deterministic_digest` over decisions that did not depend on wall-clock, plus a
count of what was excluded. `make eval-deterministic` disables both for a
bit-for-bit reproducible run; we verified identical digests across two runs at
different concurrency levels, 32 and 8. See
[ASSUMPTIONS.md, section 1.3](ASSUMPTIONS.md) — the specification asked for
something stronger than is achievable, and we say so rather than quietly
weakening the test.


## Design decisions

**Unverifiable is not the same as false.** With no source attached, groundedness
abstains — it does not score low. Abstention routes to `ANNOTATE` in low-stakes
packs and `ESCALATE` in regulated ones, and it is reported as a first-class
metric split into correct abstention, where no source existed, at 100%, and
spurious abstention, where a source was attached and we failed anyway, at 3.9%.
A system that reads "I cannot check this" as "this is wrong" drowns users in
false positives, which is the failure mode the brief warns about.

**Overlapping categories.** A fabricated detail about a real person is both a
privacy event and a hallucination. Labels are sets everywhere, and action
selection takes max severity across all triggered categories rather than the
first rule to match. Set-exact accuracy on that slice is 100%.

**Over- versus under-flagging.** There is no single accuracy figure anywhere. The
report carries full operating curves per category, and
`flags_per_100_interactions` per use case as an alert-fatigue proxy — reported
alongside `interventions_per_100`, because an annotation saying "no source was
attached" is a disclosure, not an alert.

**API-only inspection.** Nothing assumes access to logits, attention or hidden
states. A capability probe on provider initialisation decides whether logprob
entropy is available; if it is not, the system falls back to proxy signals —
retrieval overlap, entity density, hedging, self-contradiction — and records
which signal set produced every decision.

**Cached is not trusted.** A cache hit still passes the floor detectors on the
way out. The semantic cache is entity-gated: of 60 constructed near-misses, with
one identifier changed and everything else held, 0 were wrongly served. 15 were
close enough on embedding similarity that a plain semantic cache would have
served them, and were rejected on entity mismatch instead.


## Governance and audit

Every decision appends a record to a SQLite ledger where
`record_hash = sha256(prev_hash + canonical_json(record))`. Altering any record
invalidates every hash after it; `make verify` walks the chain and reports the
first break. Append-only is enforced by database triggers — `UPDATE` and
`DELETE` raise — so corrections are new records referencing the original. This
is INV-6.

Raw PII is never stored. Records carry entity types, span offsets and a salted
hash. A test asserts the ledger payload contains no source values.

Each decision also emits an OpenTelemetry span. Attribute names follow the
OpenTelemetry GenAI semantic conventions, which are still at Development status
upstream and have already been renamed once, from
`gen_ai.usage.prompt_tokens` to `gen_ai.usage.input_tokens`. The exact set this
build emits is pinned in
[ledger/otel.py, ATTRIBUTE_SET](slipstream/ledger/otel.py) and asserted in a
test. Slipstream-specific fields are namespaced `slipstream.*` rather than
invented under `gen_ai.*`.


## Tests

```bash
make test     # 139 tests
```

The ones that matter:

| Test | Asserts |
|---|---|
| `test_inv1_outbound_pii_masked_even_when_band_forced_low` | Band forced to LOW, PII still masked before emission |
| `test_inv1_floor_survives_degraded_mode` | Degradation drops everything except the floor |
| `test_inv2_never_rounds_down` | Sweeps 0.00–1.00; band is monotone and never below the raw threshold |
| `test_inv3_arms_share_one_registry_instance` | All arms hold the same registry object and fingerprint |
| `test_inv3_arm_rewrite_changes_only_scheduling_fields` | Baselines run the full detector set, so none can win on latency by checking less |
| `test_guard_for_chunk_n_overlaps_generation_of_chunk_n_plus_1` | The concurrency claim, in wall-clock |
| `test_backpressure_degrades_rather_than_stalls` | Slow guards shed work; the stream never stalls |
| `test_tampering_with_a_payload_is_detected` | Rewrite a row directly, `verify` finds it |
| `test_two_seeded_runs_produce_identical_decision_digests` | Determinism, under `--deterministic` |
| `test_offline_end_to_end_needs_no_network_or_key` | Patches `socket.connect` to raise — INV-5 enforced, not asserted |
| `test_conformal_bound_holds_in_expectation_over_calibration_draws` | The bound, tested the way conformal actually states it |
| `test_reviewer_disagreement_moves_the_threshold_and_reports_drift` | The feedback loop end to end: overrides, refit, threshold actually moves |
| `test_ledger_never_stores_raw_pii` | Ledger payload contains entity types and salted hashes, no source values |


## Limitations

This section is not a disclaimer. These are the places we know the system
breaks, and a reviewer should weigh the results above with all of them in hand.

1. **Bias detection is policy-scoped and narrow, by choice.** It is not a
   general-purpose bias classifier. The pack declares protected attributes and
   whether this is a decision context, and the detector flags a protected
   attribute appearing inside a decision rationale — "declined because she is
   likely to take maternity leave". It will not catch representational harm,
   distributional unfairness across a population, or bias expressed without an
   explicit rationale. We would not have built a general bias classifier well at
   this scale, and a reviewer would have seen through it.
1. **The synthetic corpus is not enterprise traffic.** 2,290 items of templated
   text with seeded slot-filling. Real traffic is messier, more repetitive, and
   adversarially creative in ways a generator we wrote cannot anticipate. Every
   recall figure here is an upper bound on production.
1. **The default detector engines are not the models the specification named.**
   Section 4 called for `all-MiniLM-L6-v2`, `nli-deberta-v3-small`, `toxic-bert`
   and Presidio; those total roughly 840 MB of first-run downloads, which is
   incompatible with INV-5's requirement to run offline on a clean clone. We
   shipped deterministic pure-Python engines as the default and the named models
   as opt-in flags. The offline groundedness scorer is lexical entailment plus
   numeric and entity agreement — it catches fabricated figures well and
   paraphrased fabrications poorly. See
   [ASSUMPTIONS.md, section 1.1](ASSUMPTIONS.md).
1. **Detector latency is modelled, not measured.** The offline engines run in
   about 1 ms; a comparison against 1 ms detectors would make all three arms
   look identical and prove nothing. Each detector's wall-clock is floored at
   the section 4 CPU target for the model it stands in for: floor 6 ms, cheap
   55 ms, expensive 200 ms. The decisions are always the real engine's; only the
   timing is modelled. `SLIPSTREAM_LATENCY_PROFILE=measured` turns it off. See
   [ASSUMPTIONS.md, section 1.2](ASSUMPTIONS.md).
1. **Conformal guarantees hold under exchangeability, which production drift
   violates.** The bound says that if calibration and production traffic are
   drawn from the same distribution, the expected miss rate is at most α. A new
   product line, a new locale, or a new attacker breaks that assumption and the
   guarantee goes with it. This is why the feedback module and
   `slipstream recalibrate` exist. It is also a bound in expectation over
   calibration draws, not a promise about any single split — our own test suite
   asserts it the correct way, over 40 draws.
1. **Not every threshold could be fitted.** 7 of 15 pack-and-category pairs had
   too few positives in the corpus to fit a conformal threshold at α = 0.10 and
   fall back to the pack's configured default; they are marked `UNFITTED` in the
   report. `toxicity` has no positives at all in any pack — the section 8 corpus
   table has no toxicity slice, and we did not add one the brief did not ask
   for. One coverage check failed outright: `decision_support/injection`,
   realised miss rate 0.200 against α = 0.10 on 10 held-out positives. It is in
   the report in bold rather than suppressed.
1. **The NeMo Guardrails arm was not run.** `nemoguardrails` is not importable in
   this environment, so the fourth arm was skipped and the report says so. We did
   not approximate it. Had it run, it would have been our scheduler at NeMo's
   documented streaming defaults, `chunk_size: 200` and `context_size: 50`,
   because INV-3 requires identical detectors across arms — and the report would
   have labelled it as such rather than implying NeMo's own rails executed.
1. **The `LOGPROBS` signal path is not exercised in the reported run.** The
   replay provider declares `supports_logprobs=False`, which is the honest
   default for an enterprise API deployment. We did not synthesise logprobs onto
   synthetic responses; that would have manufactured the exact signal under test.
   Both branches are covered by unit tests instead.
1. **Band thresholds are tuned against synthetic data.** They are the weakest
   numbers in the system. INV-1 exists precisely so that getting them wrong costs
   latency rather than safety.
1. **Injection recall is 90%, not 100%.** One template family in the corpus
   evades the pattern set and the offline embedding kNN. Floor enforcement means
   a missed injection is not a leak of PII, because the PII detector is
   independent — but it is a miss, and it is in the table.


## Repository layout

```
slipstream/
  contracts.py            all data contracts; written first
  gateway/    scheduler.py · pipeline.py · app.py · providers/
  risk/       estimator.py · conversation.py · features.py
  guards/     pii · injection · groundedness · toxicity · bias · consistency
  policy/     loader.py · packs/{customer_support,internal_knowledge,decision_support}.yaml
  calibration/ conformal.py · fit.py
  actions/    matrix.py · masking.py
  ledger/     store.py · otel.py
  routing/    router.py · cache.py
  feedback/   capture.py
  eval/       harness.py · metrics.py · report.py · datasets/generate.py · results/
  demo/       scenario.py · run.py
  dashboard/  index.html
tests/                    139 tests
ASSUMPTIONS.md            every parameter we chose, and what changes if it is wrong
```

`slipstream/` is the Python package. The `Makefile`, `README.md` and
`ASSUMPTIONS.md` sit at the repository root so a clone is immediately runnable.
The generated corpus and fixtures are deliberately not committed — the generator
is, and `make data` rebuilds both deterministically from the seed recorded in the
results file.


## Troubleshooting

If `make` is not recognised on Windows, use the PowerShell equivalent. Every
target has one:

```powershell
.\make.ps1 demo
```

If `make demo` or `make eval` appears to reach the network, it should not.
INV-5 requires the default path to run on a clean clone with no API key and no
downloads. `test_offline_end_to_end_needs_no_network_or_key` patches
`socket.connect` to raise, so a regression here fails the suite rather than
silently dialling out. Check whether an optional backend environment variable is
set — `SLIPSTREAM_EMBED_BACKEND`, `SLIPSTREAM_NLI_BACKEND`,
`SLIPSTREAM_TOXICITY_BACKEND` or `SLIPSTREAM_USE_PRESIDIO` all pull models on
first use.

If an optional backend raises an import error, install the extras rather than the
core pins:

```bash
pip install -r requirements-optional.txt
```

If `make eval` takes too long, run it at 25% corpus scale:

```bash
make eval-fast
```

If `make verify` reports a break in the hash chain, the ledger has been altered
after the fact. The command prints the first broken record. Append-only is
enforced by SQLite triggers, so a break means the database file was edited
outside the application, not that a write path is buggy.

If two `make eval` runs disagree, that is expected unless you disable the
wall-clock paths. Use `make eval-deterministic` for a bit-for-bit reproducible
run, and compare the `deterministic_digest` field rather than the latency
figures.

If port 8000 is already in use, `make serve` accepts an override:

```bash
make serve PORT=8080
```


## FAQ

**Q: Is this not just what NeMo Guardrails and Bedrock Guardrails already do?**

**A:** They also verify mid-stream, and we do not claim to have invented that.
The difference is that their safety–latency trade is fixed at configuration time
and applied uniformly to all traffic. Slipstream makes it a calibrated
per-request decision, and reports a conformal risk bound instead of a tuned
guess.

**Q: What happens when the risk estimator gets it wrong?**

**A:** Nothing leaks. Prompt-injection and outbound-PII detection run inline, at
full depth, in every band including the lowest. That is INV-1, enforced in the
scheduler and asserted by
`test_inv1_outbound_pii_masked_even_when_band_forced_low`. A misjudged estimate
costs latency, not safety.

**Q: Do I need an API key, a GPU, or model downloads to run this?**

**A:** No. First-run download size on the default path is 0 bytes. Optional
heavier backends exist and are all off by default.

**Q: Why is `adaptive` slower than `fixed_async` on time-to-first-token?**

**A:** Because `fixed_async` does not prevent anything. It intercepted 100% of
outbound PII and prevented 0% of it — 245 items reached the screen. The 167 ms
that `adaptive` spends over `fixed_async` is what buys 100% prevention.

**Q: Are the accuracy figures production-realistic?**

**A:** No, and we say so. The corpus is 2,290 synthetic items of templated text.
Every recall figure is an upper bound on production. The Limitations section
lists ten places the system is known to break, including a conformal coverage
check that failed outright.

**Q: How do I check that a decision was not tampered with?**

**A:** Run `make verify`. It walks the SQLite ledger's hash chain and reports the
first break. Records store entity types, span offsets and salted hashes — never
raw PII.


## Maintainers

- Yashraj Singh Thakur — [yashraj](https://github.com/YOUR-USERNAME)
