# Assumptions

The brief gives direction, not numbers. Every parameter below was chosen by us,
not derived from data we were given. This file records what each one is, what we
set it to, why, and what would change if it were wrong — so a reviewer can
disagree with a specific number rather than with the whole system.

Anything marked **load-bearing** changes the headline result if you move it.
Anything marked *cosmetic* does not.

---

## 1. Deviations from the build spec

Three places where we did something other than what the spec said. Each is
flagged here rather than quietly worked around (§15).

### 1.1 The model stack is opt-in, not default — **load-bearing**

**Spec §4** names `all-MiniLM-L6-v2`, `nli-deberta-v3-small`, `toxic-bert`,
`presidio-analyzer` and spaCy `en_core_web_sm`. **INV-5** requires `make demo`
and `make eval` to run with no network access on a clean clone. These cannot both
be true: those models total roughly 840 MB of first-run downloads.

We resolved it in favour of INV-5. Every model-backed detector has a
deterministic pure-Python default that needs no download:

| Concern | Default (offline) | Opt-in (`requirements-optional.txt`) |
|---|---|---|
| Embeddings | hashed word/bigram/char-4gram vectors, 256-d | `SLIPSTREAM_EMBED_BACKEND=st` |
| Groundedness | lexical entailment + numeric/entity agreement | `SLIPSTREAM_NLI_BACKEND=nli` |
| Toxicity | severity-weighted lexicon with negation discount | `SLIPSTREAM_TOXICITY_BACKEND=bert` |
| PII | regex + checksum validation + name gazetteer | `SLIPSTREAM_USE_PRESIDIO=1` |

**First-run download size on the default path: 0 bytes.**

The scheduling, the policy packs, the calibration and the action matrix are
identical either way — only detector quality changes. `results.json` records
which backend produced every number under `run.backends`, so no figure is ever
ambiguous about what generated it.

What this costs: the offline embedding backend is lexical, so injection kNN is
weaker against novel paraphrases, and the lexical groundedness scorer will miss
paraphrased fabrications that a real NLI model would catch. Detector recall in
`report.md` should be read as *"what these detectors achieve"*, not *"what the
§4 stack would achieve"*.

### 1.2 Detector wall-clock is modelled, not measured — **load-bearing**

The offline engines run in about 1 ms. The transformer models they stand in for
run in tens to low hundreds of milliseconds on CPU. Measuring the three-arm
comparison against 1 ms detectors would make all three arms look identical and
would prove nothing — the entire question is what happens when verification is
*expensive*.

So each detector's wall-clock is floored at the §4 CPU target for the model it
substitutes for:

| cost class | modelled floor | stands in for |
|---|---|---|
| `floor` | 6 ms | presidio-analyzer + spaCy NER over a chunk |
| `cheap` | 55 ms | `nli-deberta-v3-small`, §4 target "< 80 ms per chunk" |
| `expensive` | 200 ms | multi-sample agreement over 3 completions |

**The decisions are always the real engine's. Only the timing is modelled.** Set
`SLIPSTREAM_LATENCY_PROFILE=measured` for raw measured latency; the profile in
force is recorded in `results.json`.

One exception, and it matters: an abstention that means *"there was no source to
check against"* is **not** charged the modelled latency, because in a real
deployment the NLI model is never invoked in that case. Charging unverifiable
traffic for a call that never happens would have inflated the latency of exactly
the slice §9 is about.

### 1.3 `make eval` is not bit-for-bit reproducible by default — **load-bearing**

**Spec §13** asks that two seeded eval runs produce identical `results.json`. As
specified this is not achievable, and we think the spec is wrong here rather than
the implementation.

Two mechanisms make a decision depend on wall-clock, and both are *real
enforcement* that the spec itself requires:

* **backpressure degradation** (§7.1) — guards that lose the race to generation
  for `k` consecutive chunks drop the policy to floor-only;
* **guard budget timeouts** (§7.6 `guard_budget_ms`) — a detector that overruns
  its budget abstains.

Whether either fires depends on machine load, and both change which detectors
contributed to the outcome. A run that was bit-for-bit reproducible would be one
with no load-shedding and no budget enforcement.

**What we did instead.** `results.json` carries a `deterministic_digest` over the
decisions that did *not* depend on wall-clock, plus a count of the items excluded
for that reason and the rates of both mechanisms. `--deterministic` disables both
and gives a fully reproducible run; that is what the determinism test asserts
against. Latency figures are never part of any digest — they are machine
dependent, and asserting equality on them would be a test that passes by luck.

---

## 2. Risk and calibration parameters

| Parameter | Value | Why | If it were wrong |
|---|---|---|---|
| `alpha` (target miss rate) | `0.10` | **Load-bearing.** No basis in the brief; 10% is the conventional starting point for conformal risk control and keeps the thresholds inside the range where the corpus has enough positives to fit. | Lower alpha lowers every threshold, raising recall and flag rate together. The operating curves in `report.md` show the whole trade so you can pick a different point. |
| `beta` (escalation budget) | `0.02` support, `0.04` knowledge, `0.06` decision support | The brief gives `0.06` for decision support explicitly. The other two are ours: a consumer-facing support desk tolerates far less human review per interaction than a regulated workflow. | Directly changes the "escalation vs budget" verdict. Reported against **benign** traffic only — see §4 below. |
| Calibration/test split | 50/50, stratified by (slice, label set), seed `1729` | Equal halves maximise the smaller side, which is the binding constraint on whether a category can be fitted at all. Multi-turn sessions are kept whole; splitting a conversation would destroy the ratchet signal. | A smaller calibration half pushes more (pack, category) pairs into `UNFITTED`. |
| Degenerate fits fall back to pack defaults | yes | **Load-bearing.** When `floor(alpha*(n+1)) < 1`, no threshold above 0 meets alpha. Publishing `tau = 0` would satisfy the bound by flagging 100% of traffic. We report the pair as `UNFITTED` and keep the pack's configured default. | Publishing tau=0 makes the system flag everything. We found this the hard way — the first eval run blocked every request. |
| Band boundaries | medium `0.32-0.34`, high `0.62-0.66` (per pack) | Chosen so the benign slices land in LOW and the risky slices in MEDIUM/HIGH on this corpus. **This is the weakest number in the system**: it is tuned against synthetic data. | Shifts the whole latency/safety trade. The floor (INV-1) is unaffected by band, which is why INV-1 exists. |
| `uncertainty_margin` | `0.06`-`0.08` | The brief gives `0.08` for decision support. Tighter margins for lower-stakes packs. | Widening it moves more traffic up a band and more scores into ANNOTATE. |
| Estimator self-confidence < `0.5` rounds up | yes | INV-2's spirit: an estimator that does not trust itself must not be the thing that chooses the cheaper policy. | Removing it lets short, signal-free prompts sit in LOW. |

---

## 3. Scheduling parameters

| Parameter | Value | Why | If it were wrong |
|---|---|---|---|
| `FLOOR_HOLDBACK_TOKENS` | `16` | **Load-bearing.** INV-1 says the floor runs *inline* in every band, so `holdback_tokens: 0` cannot mean "emit unchecked". It means "as shallow as the floor allows". 16 tokens is roughly one clause — enough to cut at a boundary without holding a visible amount of text. | Smaller and chunks get too short to contain a complete identifier; larger and LOW-band TTFT rises. The `fixed_async` arm is the only configuration exempt, because being unable to prevent a leak is the property it exists to measure. |
| Backpressure `k` | `3` | Three consecutive lost races is enough to distinguish sustained overload from one slow chunk. | Lower degrades too eagerly; higher lets the buffer grow under load. Measured incidence is in the headline table. |
| `CHARS_PER_TOKEN` | `4` | Converts a token-denominated holdback into a character buffer depth without a tokeniser call. The scheduler must not spend milliseconds deciding where to spend milliseconds. | *Cosmetic.* Affects chunk sizes by a constant factor. |
| Holdback depths | 0/16 · 48-80 · 192-256 tokens | Roughly: one clause, one sentence, one paragraph. The `decision_support` values (64 / 256) are the brief's. | Deeper buffers give guards a larger window to win in — visible in the per-band overlap table. |
| Simulated generation speed | 5 ms/token | ~200 tok/s, a realistic mid-size hosted model. Set with `SLIPSTREAM_TOKEN_MS`. | **Load-bearing.** This is the denominator of the guard-vs-generation race. A faster model shrinks the window guards have to win in; the per-band overlap table is the place to look. |
| `nemo_default` chunking | 200 / 50 | NeMo Guardrails' documented streaming defaults. Only used if the arm runs at all. | n/a — the arm was skipped, see README. |

---

## 4. Metric definitions we had to choose

The brief names metrics but not their definitions. These choices change what the
numbers mean.

* **Escalation budget is checked against benign traffic, not the whole corpus.**
  A pack's `beta` is a target share of *production* traffic. Our corpus is
  deliberately risk-enriched (35-58% risky depending on use case), so the
  whole-corpus escalation rate is not comparable to `beta` and reporting it that
  way would manufacture a failure. Both figures and the risky share are printed
  side by side.

* **"Flags" and "interventions" are reported separately.** An `ANNOTATE` that
  says *"no source was attached, so this was not checked"* is an honest
  disclosure, not an alert. Counting it as one would overstate alert fatigue by
  the size of the unverifiable slice. `flags_per_100` counts every non-ALLOW
  outcome; `interventions_per_100` counts only MASK / REGENERATE / ESCALATE /
  BLOCK.

* **"PII interception" and "PII masked before emission" are different metrics.**
  Interception means the system detected it. Masked-before-emission means the
  user never saw it. The gap between the two *is* the `fixed_async` arm's failure
  mode, and collapsing them would hide the entire result.

* **A "leak" is content that would have been withheld or transformed had it been
  held back** — not "the final action was MASK". Keying it on the action label
  missed every case where a higher-severity action outranked MASK while the PII
  still reached the user.

* **A regenerated request records `REGENERATE` as its action**, not the clean
  retry's own outcome, and the first attempt's leaks and masking carry forward.
  A clean retry does not retroactively un-display what was already on screen.

* **Guard-vs-generation overlap excludes chunks under a synchronous policy and
  the final chunk of every response.** Neither had a race to win, and counting
  them as wins would inflate the headline metric with requests that never
  exercised the mechanism.

* **Token cost: the guard term is zero.** Every detector here is a local CPU
  model, not an LLM judge, so verification costs no tokens. A design that used an
  LLM as a guard would have a very different number and this comparison would not
  carry over. Prices are illustrative; only the cheap/frontier *ratio* matters.

---

## 5. Corpus assumptions

* **Slice sizes are the brief's §8 table exactly**, at `--scale 1.0`.
* **Labels are sets.** The overlapping slice carries both `privacy` and
  `hallucination`, because a fabricated detail about a real person is both.
* **The unverifiable slice carries an empty label set** plus
  `expects_abstention=True`. Scoring those items as violations would reward the
  behaviour §9 warns against.
* **Two slices were spread across all three packs** (`pii_leak`,
  `ungrounded_with_context`). A (pack, category) pair with no positives cannot be
  conformally fitted, and the first run left `customer_support/hallucination` and
  `decision_support/privacy` with zero.
* **`toxicity` and `bias` still have no positives in some packs.** The §8 table
  has no toxicity slice, and bias is decision-support only. Those pairs are
  reported `UNFITTED` and fall back to pack defaults. We did not add slices the
  brief did not ask for; the gap is visible in the calibration table.
* **This is templated text with seeded slot-filling. It is not enterprise
  traffic.** Detector recall measured here is an upper bound on production.

---

## 6. Things we deliberately did not assume

* **No logprobs.** The replay provider declares `supports_logprobs=False`, which
  is the honest default for an enterprise API deployment (§7.4). We did not
  synthesise logprobs onto synthetic responses — that would have manufactured the
  exact signal under test. The `LOGPROBS` signal mode is therefore **not
  exercised** in the reported run; the capability probe and both branches are
  covered by unit tests instead.

* **No general-purpose bias classifier.** Bias is a policy-scoped check: the pack
  declares protected attributes and whether this is a decision context, and the
  detector flags a protected attribute appearing inside a decision rationale.
  Narrow, and honest about being narrow.

* **No NeMo Guardrails arm.** `nemoguardrails` is not importable in this
  environment. The arm was skipped and said so rather than approximated. Had it
  run, it would have been *our* scheduler at NeMo's documented chunking defaults
  — required for INV-3 — and the report would have labelled it as such rather
  than implying NeMo's own rails executed.
