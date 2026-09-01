# Slipstream

**A model-agnostic streaming gateway that decides, per request, how much
verification a response deserves — and schedules that verification to run
concurrently with token generation instead of in front of it.**

Accenture Innovation Challenge 2026 · Round 2 · PS1 ControlPlane.ai

---

## The thesis, in three sentences

Streaming guardrails already exist: NVIDIA NeMo Guardrails and AWS Bedrock
Guardrails both verify model output mid-stream, and we are not claiming to have
invented that. What they share is that the safety–latency trade is **fixed at
configuration time and applied uniformly to all traffic** — one chunk size, one
buffer depth, one set of checks, whether the request is "what are your opening
hours" or "summarise this employee's compensation record". Slipstream makes that
setting a **calibrated per-request decision**: a sub-15 ms prompt-side estimator
picks a verification band, the scheduler holds back exactly as much of the stream
as that band requires, and split-conformal calibration turns the resulting
thresholds into a stated risk bound instead of a tuned guess.

The obvious attack on this idea is "what happens when your risk estimator is
wrong?" The answer is **INV-1**: prompt-injection and outbound-PII detection run
inline, at full depth, in *every* band including the lowest. A misjudged risk
estimate can cost latency. It cannot open the leak path. That is enforced in the
scheduler and asserted in the test suite, not promised in a README.

---

## The result

Full table: [`slipstream/eval/results/report.md`](slipstream/eval/results/report.md) ·
raw: [`results.json`](slipstream/eval/results/results.json)

All three arms run the **same detectors at the same thresholds against the same
provider** (INV-3 — the harness asserts the detector-registry fingerprint is
identical across arms and records it in the results). Only the scheduling policy
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

**Adaptive closes 64% of the TTFT gap between the safe and fast baselines while
holding the safe baseline's interception rate and recall exactly.**

Two things are worth more than the headline:

- **`fixed_async` intercepts 100% of outbound PII and prevents 0% of it.** It
  detects every leak — after the tokens are on the screen. Interception and
  prevention are different metrics, and the gap between them is the entire
  argument for hold-back scheduling.
- Under `--deterministic`, **`adaptive` and `fixed_sync` produce the same
  decision digest** — bit-identical decisions on every item. Adaptive is not
  approximating the safe baseline on this corpus; it is reproducing it, at 40% of
  its time-to-first-token.

### Reproducing it

```bash
make eval          # ~2 min, offline, regenerates the corpus, the calibration and both files
```

Seed `1729`, corpus 2,290 items (1,145 held-out test items per arm), backends and
platform recorded in `results.json` under `run`. The exact command is printed in
the report as `run.reproduce`.

**On determinism:** decisions are reproducible; wall-clock is not. Two mechanisms
make a decision depend on machine load — backpressure degradation and
guard-budget timeouts — and both are real enforcement the spec requires. The
results carry a `deterministic_digest` over decisions that did *not* depend on
wall-clock, plus a count of what was excluded. `make eval-deterministic` disables
both for a bit-for-bit reproducible run; we verified identical digests across two
runs at different concurrency levels (32 and 8). See
[ASSUMPTIONS.md §1.3](ASSUMPTIONS.md) — the spec asked for something stronger
than is achievable, and we say so rather than quietly weakening the test.

---

## Quickstart

No API key. No network access. No model downloads. **First-run download size on
the default path: 0 bytes.**

```bash
git clone <repo> && cd slipstream
pip install -r requirements.txt      # pure-Python + small wheels only
make demo                            # the A/B scenario from the video
```

<details>
<summary>Windows (no <code>make</code>)</summary>

```powershell
py -3.11 -m pip install -r requirements.txt
.\make.ps1 demo
```
Every Makefile target has a `.\make.ps1 <target>` equivalent.
</details>

| Command | What it does |
|---|---|
| `make demo` | Offline A/B: the same request through `fixed_async` and `adaptive`, side by side |
| `make serve` | Gateway + live dashboard at `http://127.0.0.1:8000/` |
| `make eval` | Regenerates the corpus, refits calibration, rebuilds the results table |
| `make eval-fast` | The same at 25% corpus scale (~30 s) |
| `make test` | 139 tests |
| `make verify` | Walks the ledger hash chain, reports the first break |
| `make info` | Prints active backends, detectors and the registry fingerprint |

### The demo

An internal finance agent is asked to summarise a compensation record.

- **Left (`fixed_async`)** — response streams immediately. The salary figure, the
  payroll account, the PAN and the contact details all reach the screen. The
  violation is flagged afterwards, and the pane keeps showing what was displayed
  before the correction — erasing a response does not un-display it.
- **Right (`adaptive`)** — the prompt scores HIGH, the buffer deepens to 224
  tokens, and every one of those entities is masked *before the first chunk is
  released*.

Both panes show live TTFT and added-latency counters, and the decision record for
each with its place in the hash chain visible. `make serve` gives the same thing
in a browser with server-sent events.

---

## Architecture

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

Chunk *N* is held. Its detectors run **while the provider generates chunk
*N+1***. If the verdict is clean by the time chunk *N+1* is cut, chunk *N* is
released and verification cost nothing in wall-clock. If it is dirty, chunk *N*
is never emitted.

That race is what buys the latency back, so it is measured rather than asserted:
`pct_chunks_guard_finished_first` is a headline metric, broken out per band
(shallow LOW hold-back gives detectors a small window; deep HIGH hold-back gives
them a large one), and a test asserts the wall-clock overlap directly.

---

## What is actually here

| Piece | Where | Note |
|---|---|---|
| Data contracts | [`contracts.py`](slipstream/contracts.py) | Written first. Multi-label throughout; abstention is a first-class state |
| Hold-back scheduler | [`gateway/scheduler.py`](slipstream/gateway/scheduler.py) | The differentiator. Three regimes, one code path |
| Tier-0 estimator | [`risk/estimator.py`](slipstream/risk/estimator.py) | p95 measured under 15 ms; no LLM in the path, and a test that guards against one being added |
| Conversation ratchet | [`risk/conversation.py`](slipstream/risk/conversation.py) | Band floor rises across turns and decays on a half-life; write-scoped tools force HIGH |
| Detectors | [`guards/`](slipstream/guards/) | pii · injection (floor) · groundedness · toxicity · bias (cheap) · consistency (expensive) |
| Policy packs | [`policy/packs/`](slipstream/policy/packs/) | Three use cases × geographies, hot-reloaded from YAML |
| Conformal calibration | [`calibration/conformal.py`](slipstream/calibration/conformal.py) | With a mandatory empirical coverage check |
| Action matrix | [`actions/matrix.py`](slipstream/actions/matrix.py) | Max severity across triggered categories |
| Hash-chained ledger | [`ledger/store.py`](slipstream/ledger/store.py) | Append-only enforced by SQLite triggers, not convention |
| Three-arm harness | [`eval/harness.py`](slipstream/eval/harness.py) | Score → fit → run → probe → report |
| Corpus generator | [`eval/datasets/generate.py`](slipstream/eval/datasets/generate.py) | Committed; the corpus is not (§8) |
| Dashboard | [`dashboard/index.html`](slipstream/dashboard/index.html) | One file, vanilla JS, SSE, no build step |

### Things that are easy to get wrong, and what we did

**Unverifiable is not the same as false.** With no source attached, groundedness
**abstains** — it does not score low. Abstention routes to `ANNOTATE` in
low-stakes packs and `ESCALATE` in regulated ones, and it is reported as a
first-class metric split into *correct* abstention (no source existed: 100%) and
*spurious* abstention (a source was attached and we failed anyway: 3.9%). A
system that reads "I cannot check this" as "this is wrong" drowns users in false
positives, which is the failure mode the brief warns about.

**Overlapping categories.** A fabricated detail about a real person is both a
privacy event and a hallucination. Labels are sets everywhere; action selection
takes max severity across all triggered categories rather than the first rule to
match. Set-exact accuracy on that slice: **100%**.

**Over- vs under-flagging.** No single accuracy figure anywhere. The report
carries full operating curves per category, and `flags_per_100_interactions` per
use case as an alert-fatigue proxy — reported alongside
`interventions_per_100`, because an annotation saying "no source was attached"
is a disclosure, not an alert.

**API-only inspection.** No assumption of logits, attention or hidden states. A
capability probe on provider init decides whether logprob entropy is available;
if not, the system falls back to proxy signals (retrieval overlap, entity
density, hedging, self-contradiction) and records which signal set produced every
decision.

**Cached is not trusted.** A cache hit still passes the floor detectors on the
way out. And the semantic cache is entity-gated: of 60 constructed near-misses
(one identifier changed, everything else held), **0 were wrongly served**; 15
were close enough on embedding similarity that a plain semantic cache would have
served them and were rejected on entity mismatch instead.

---

## Limitations

This section is not a disclaimer. These are the places we know the system breaks,
and a reviewer should weigh the results above with all of them in hand.

1. **Bias detection is policy-scoped and narrow — by choice.** It is not a
   general-purpose bias classifier. The pack declares protected attributes and
   whether this is a decision context, and the detector flags a protected
   attribute appearing inside a decision rationale ("declined *because she is
   likely to take maternity leave*"). It will not catch representational harm,
   distributional unfairness across a population, or bias expressed without an
   explicit rationale. We would not have built a general bias classifier well at
   this scale, and a reviewer would have seen through it.

2. **The synthetic corpus is not enterprise traffic.** 2,290 items of templated
   text with seeded slot-filling. Real traffic is messier, more repetitive, and
   adversarially creative in ways a generator we wrote cannot anticipate. **Every
   recall figure here is an upper bound on production.**

3. **The default detector engines are not the models the spec named.** §4 called
   for `all-MiniLM-L6-v2`, `nli-deberta-v3-small`, `toxic-bert` and Presidio;
   those total ~840 MB of first-run downloads, which is incompatible with INV-5's
   requirement to run offline on a clean clone. We shipped deterministic
   pure-Python engines as the default and the named models as opt-in flags
   (`requirements-optional.txt`). The offline groundedness scorer is lexical
   entailment plus numeric/entity agreement — it catches fabricated *figures*
   well and paraphrased fabrications poorly. See
   [ASSUMPTIONS.md §1.1](ASSUMPTIONS.md).

4. **Detector latency is modelled, not measured.** The offline engines run in
   ~1 ms; a comparison against 1 ms detectors would make all three arms look
   identical and prove nothing. Each detector's wall-clock is floored at the §4
   CPU target for the model it stands in for (floor 6 ms, cheap 55 ms, expensive
   200 ms). **The decisions are always the real engine's; only the timing is
   modelled.** `SLIPSTREAM_LATENCY_PROFILE=measured` turns it off.
   [ASSUMPTIONS.md §1.2](ASSUMPTIONS.md).

5. **Conformal guarantees hold under exchangeability, which production drift
   violates.** The bound says: if calibration and production traffic are drawn
   from the same distribution, the expected miss rate is at most α. A new product
   line, a new locale, or a new attacker breaks that assumption and the guarantee
   goes with it. This is why `feedback/` and `slipstream recalibrate` exist. It is
   also a bound *in expectation over calibration draws*, not a promise about any
   single split — our own test suite asserts it the correct way, over 40 draws.

6. **Not every threshold could be fitted.** 7 of 15 (pack, category) pairs had too
   few positives in the corpus to fit a conformal threshold at α = 0.10 and fall
   back to the pack's configured default; they are marked `UNFITTED` in the
   report. `toxicity` has no positives at all in any pack — the §8 corpus table
   has no toxicity slice, and we did not add one the brief did not ask for. **One
   coverage check failed outright**: `decision_support/injection`, realised miss
   rate 0.200 against α = 0.10 on 10 held-out positives. It is in the report in
   bold rather than suppressed.

7. **The NeMo Guardrails arm was not run.** `nemoguardrails` is not importable in
   this environment, so the fourth arm was skipped and the report says so. We did
   not approximate it. Had it run it would have been *our* scheduler at NeMo's
   documented streaming defaults (`chunk_size: 200`, `context_size: 50`) — INV-3
   requires identical detectors across arms — and the report would have labelled
   it as such rather than implying NeMo's own rails executed.

8. **The `LOGPROBS` signal path is not exercised in the reported run.** The
   replay provider declares `supports_logprobs=False`, which is the honest
   default for an enterprise API deployment. We did not synthesise logprobs onto
   synthetic responses; that would have manufactured the exact signal under test.
   Both branches are covered by unit tests instead.

9. **Band thresholds are tuned against synthetic data.** They are the weakest
   numbers in the system. INV-1 exists precisely so that getting them wrong costs
   latency rather than safety.

10. **Injection recall is 90%, not 100%.** One template family in the corpus
    evades the pattern set and the offline embedding kNN. Floor enforcement means
    a missed injection is not a leak of PII — the PII detector is independent —
    but it is a miss, and it is in the table.

---

## Governance and audit

Every decision appends a record to a SQLite ledger where
`record_hash = sha256(prev_hash + canonical_json(record))`. Altering any record
invalidates every hash after it; `make verify` walks the chain and reports the
**first** break. Append-only is enforced by database triggers — `UPDATE` and
`DELETE` raise — so corrections are new records referencing the original (INV-6).

**Raw PII is never stored.** Records carry entity *types*, span offsets and a
salted hash. There is a test asserting the ledger payload contains no source
values.

Each decision also emits an OpenTelemetry span. Attribute names follow the
OpenTelemetry **GenAI semantic conventions**, which are still *Development*
status upstream and have already been renamed once
(`gen_ai.usage.prompt_tokens` → `gen_ai.usage.input_tokens`). The exact set this
build emits is pinned in
[`ledger/otel.py:ATTRIBUTE_SET`](slipstream/ledger/otel.py) and asserted in a
test; Slipstream-specific fields are namespaced `slipstream.*` rather than
invented under `gen_ai.*`.

---

## Configuration

Behaviour varies by use case and geography through YAML, not code
([`policy/packs/`](slipstream/policy/packs/)), hot-reloaded on change.

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

Optional backends, all off by default and all documented in
[`requirements-optional.txt`](requirements-optional.txt):

```bash
SLIPSTREAM_EMBED_BACKEND=st        # sentence-transformers/all-MiniLM-L6-v2
SLIPSTREAM_NLI_BACKEND=nli         # cross-encoder/nli-deberta-v3-small
SLIPSTREAM_TOXICITY_BACKEND=bert   # unitary/toxic-bert
SLIPSTREAM_USE_PRESIDIO=1          # presidio-analyzer + spaCy
SLIPSTREAM_PROVIDER=openai         # any OpenAI-compatible /v1/chat/completions
```

---

## Tests

```bash
make test     # 139 tests
```

The ones that matter:

| Test | Asserts |
|---|---|
| `test_inv1_outbound_pii_masked_even_when_band_forced_low` | Band forced to LOW, PII still masked before emission |
| `test_inv1_floor_survives_degraded_mode` | Degradation drops everything *except* the floor |
| `test_inv2_never_rounds_down` | Sweeps 0.00–1.00; band is monotone and never below the raw threshold |
| `test_inv3_arms_share_one_registry_instance` | All arms hold the same registry object and fingerprint |
| `test_inv3_arm_rewrite_changes_only_scheduling_fields` | Baselines run the *full* detector set, so none can win on latency by checking less |
| `test_guard_for_chunk_n_overlaps_generation_of_chunk_n_plus_1` | The concurrency claim, in wall-clock |
| `test_backpressure_degrades_rather_than_stalls` | Slow guards shed work; the stream never stalls |
| `test_tampering_with_a_payload_is_detected` | Rewrite a row directly, `verify` finds it |
| `test_two_seeded_runs_produce_identical_decision_digests` | Determinism, under `--deterministic` |
| `test_offline_end_to_end_needs_no_network_or_key` | Patches `socket.connect` to raise — INV-5 enforced, not asserted |
| `test_conformal_bound_holds_in_expectation_over_calibration_draws` | The bound, tested the way conformal actually states it |
| `test_reviewer_disagreement_moves_the_threshold_and_reports_drift` | The feedback loop end to end: overrides → refit → threshold actually moves |
| `test_ledger_never_stores_raw_pii` | Ledger payload contains entity types and salted hashes, no source values |

---

## Repository

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

`slipstream/` is the Python package; `Makefile`, `README.md` and `ASSUMPTIONS.md`
sit at the repository root so a clone is immediately runnable.
