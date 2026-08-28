# Investigating Forward Planning Behaviors in Smaller LLMs through Transcoder Circuit Tracing

## Overview

This project investigates whether transformer language models engage in **advance planning** when generating rhyming couplets. Specifically: when a model generates the second line of a couplet, does it "decide" on the rhyme word before it starts writing (a *planning* feature), or does it only settle on it once it's already writing the word (an *execution* feature)?

The pipeline runs across three model sizes — Gemma-3 270M, 1B, and 4B — using [Gemma-Scope-2](https://huggingface.co/google/gemma-scope-2) transcoders to replace MLP layers with interpretable sparse features. Attribution graphs are built for every generated token across an 11-prompt set, feature timelines are extracted and classified as planning or execution, and causal interventions (feature suppression) measure which features are causally load-bearing for the rhyme.

---

## Glossary

**Prompt selection** (`tools/prompt_set.py`)

| Term | Meaning | Defined by |
|---|---|---|
| rime | A word from its primary-stressed vowel to the end. | `rhyming_part_primary()` |
| onset clause | Words with identical consonants before the stressed vowel are not rhymes (*rime riche*). Never fires on a vowel-initial word. | `shares_onset()` |
| `rhymes_all` | Rhyme count over all of CMUdict. Primary pronunciations, alphabetic entries, onset clause applied, self excluded. Reported per prompt. | `family_size()` |
| `rhymes_common` | The same count restricted to the wordfreq top-10,000 band. | `banded_family_size_all()` |
| rhyme family | Words sharing one rime. Prompts use one per family. | `candidate_rows()` |
| affixal | Every available rhyme is a prefix or suffix of the word, so the couplet can be completed by affixation rather than phonological retrieval. Over-inclusive by design; flags for manual inspection, never filters. | `is_affixal()` |
| difficulty | hard = 1–9 rhymes, medium = 10–99, easy = 100+. The integer part of `log10(rhymes_all)`; descriptive only, never used to select, and derived on demand rather than stored in `prompt_set.json`. | `difficulty_label()` |
| grid | 10 prompts at `rhymes_all` targets evenly spaced in log10 from 1 to 115 (`1, 2, 3, 5, 8, 14, 24, 40, 68, 115`), each an exact `rhymes_all` match. | `GRID_TARGETS`, `grid_coverage()` |
| `original` | An 11th prompt reproducing the paper's original "grab it"/"rabbit" example verbatim, `rhymes_all: null` since it's a cross-word near-rhyme the metric can't score. Kept as a replication anchor, exempt from the grid. | — |

**Output judgement** (`experiment/rhyme_labels.py`)

| Term | Meaning | Defined by |
|---|---|---|
| near rhyme | Primary stressed vowel matches, full rime differs. | `label()` |
| line echo | The generated line reproduces the prompt's line, compared as word sequences. | `line_echo` |
| line overlap | Fraction of the prompt line's distinct words reappearing in the generated line. Reported as a number, never thresholded. | `line_overlap()` |
| rhyme-word echo | The generated line ends on the prompt's line-ending word. | `rhyme_word_echo` |
| inflected repeat | The rhyme word is not the target but has it as a prefix (grab/grabbed). Prefix only — a suffix rule would fire on ordinary rhymes. | `inflected_repeat()` |
| looped | The run never emitted a stop token. `rhyme_step` is `null`, so `tracing.py` skips it — planning vs execution is only meaningful relative to a rhyme step. | `rhyme_labels.py` |

**Feature analysis** (`experiment/tracing.py`)

| Term | Meaning | Defined by |
|---|---|---|
| planning feature | Influence peaks before the rhyme step. | `peak_step < rhyme_step` |
| execution feature | Influence peaks at or after the rhyme step. Excluded from measured populations — its position can fall past the end of the measurement sequence. | the complement |
| sustain ratio | Influence at the rhyme step ÷ peak influence. | `sustain_ratio` |
| rhyme-circuit candidate | A planning feature also active at the rhyme step, at or above the configured influence/percentile/sustain cutoffs. | `candidate_keys()` |
| `logit_drop` | Drop in the rhyme token's logit when a feature is suppressed at its recorded position. The headline suppression statistic (mean over top k=10) — unlike `prob_drop`, it doesn't saturate once the target probability is near zero. | `downstream_effects_addon.py` |
| last-layer exclusion | Features in the model's final layer can't affect any later position (no attention layer follows the last MLP), so their suppression effect is always exactly zero. Excluded from every measured population. | `feature_stats(n_layers=...)` |

Code documentation was generated with Claude (Anthropic).

---

## Pipeline Overview

```
generation-gemma-3-{270m,1b,4b}.py
        │  greedy-decodes the second line token-by-token, then replays the
        │  finished trace through circuit_tracer.attribute()
        │  outputs: experiment/graphs/gemma-3-{size}-it/{slug}/step-NN-{token}.json
        ▼
rhyme_labels.py
        │  GPU-free. Labels each continuation's last content word against
        │  its prompt's target (rhyme/near_rhyme/repetition/none/oov/degenerate)
        │  outputs: experiment/rhyme_labels.json
        ▼
tracing.py  (+ tracing-{270m,1b,4b}.py config shims)
        │  all GPU work: builds feature timelines, classifies planning vs
        │  execution features, measures candidate/superset/near-miss/random
        │  populations with causal suppression
        │  outputs: experiment/tracing/circuit_tracing_results_{size}_{slug}.json
        ▼
threshold_sensitivity.py
        │  GPU-free. Sweeps influence/percentile/sustain-ratio cutoffs
        │  against tracing.py's measured populations
        │  outputs: experiment/threshold/threshold_sensitivity_{size}_{slug}.json
        ▼
comparing.py
        │  aggregates per-prompt results across sizes, split by rhyme label,
        │  regresses planning measures on log10(rhymes_all)
        │  outputs: terminal report + experiment/comparison.json
```

All five stages key off the same 11-prompt set in `tools/prompt_set.json` and use the `-it` (instruction-tuned) checkpoints.

---

## Step 1 — Graph Generation

**Files:** `generation-gemma-3-270m.py`, `generation-gemma-3-1b.py`, `generation-gemma-3-4b.py`

### What they do

Each script loads a Gemma-3 model patched with Gemma-Scope-2 transcoders, then runs two passes per prompt, back-to-back but strictly separated:

**Generation pass** — greedy-decodes the second line one token at a time, stopping at punctuation or EOS. Every generated token is recorded with its token ID, string, and probability. This finishes completely before any attribution runs, so nothing in the attribution path can affect what the model wrote.

**Attribution pass** — replays the finished token trace through `circuit_tracer.attribute()`, producing one attribution graph per generated token: which transcoder features at which layer and position were most responsible for predicting that token.

A resume guard skips any prompt slug whose output directory is non-empty — it only checks for *any* file present, not completeness, so a partially-written directory needs to be cleared before re-running it.

### Prompts

All 11 prompts in `tools/prompt_set.json` are run per size (10 on the log10 `rhymes_all` grid plus the `original` replication anchor) — see the Glossary above.

### Key parameters

| Parameter | Value | Effect |
|---|---|---|
| `max_n_logits` | 5 | Maximum number of output logits to attribute from |
| `desired_logit_prob` | 0.95 | Stops attributing once this probability mass is covered |
| `max_feature_nodes` | 1028 | Maximum transcoder features included in the graph |
| `batch_size` | 8 | Batch size during attribution |
| `offload` | `"disk"` | Offloads intermediate tensors to disk to save GPU memory |

### Outputs

Graph files are written to `experiment/graphs/gemma-3-{size}-it/{slug}/step-NN-{token}.json`. Each file contains a list of nodes (transcoder features, residual stream positions, logit targets) and edges (influence weights between them), plus `metadata.prompt` and `metadata.prompt_tokens` used by later stages.

### Model/transcoder details

| Size | Model | Transcoder repo | Layers |
|---|---|---|---|
| 270M | `google/gemma-3-270m-it` | `gemma-scope-2-270m-it` | 18 |
| 1B | `google/gemma-3-1b-it` | `gemma-scope-2-1b-it` | 26 |
| 4B | `google/gemma-3-4b-it` | `gemma-scope-2-4b-it` | 34 |

All transcoders use width `16k`, L0 `small`, and hook points `hook_resid_mid` (input) → `hook_mlp_out` (output).

---

## Step 2 — Rhyme Labelling

**File:** `rhyme_labels.py`

GPU-free. Reconstructs each continuation from the graphs' `metadata.prompt` (stripping the chat prefix), takes the **last content word** as the rhyme unit (via `nltk.pos_tag`), and labels it against the prompt's target using the CMUdict layer in `tools/prompt_set.py`: `rhyme` / `near_rhyme` / `repetition` / `none` / `oov` / `degenerate`.

Also emits natural-match diagnostics: `line_overlap`, `looped`, and `inflected_repeat`. A looped run gets `rhyme_step: null`, which `tracing.py` treats as unmeasurable — see Glossary.

Writes `experiment/rhyme_labels.json`, which is the sole config source for Step 3 (no hardcoded rhyme token/step constants anywhere downstream).

Without the NLTK `wordnet` corpus, near-rhyme detection falls back to exact-match repetition detection (`stab`/`stabbed` would score as a rhyme rather than a repetition) and the script prints a note about this at the end of the run. See "Environment setup" below.

---

## Step 3 — Feature Tracing & Causal Intervention

**Files:** `tracing.py` (shared implementation), `tracing-270m.py` / `tracing-1b.py` / `tracing-4b.py` (13-line config shims that just call `tracing.main()` with a `SizeConfig`)

All GPU work in the experiment lives here — one model load per size measures every population any downstream analysis needs.

```
python experiment/tracing-4b.py                     # analysis + interventions
python experiment/tracing-4b.py --no-interventions   # GPU-free half only
python experiment/tracing-4b.py --slugs realm ten    # selected prompts
```

### Positions, not steps

The second element of an intervention tuple is a **token index into the tokenized input**, not a generation step. Suppression positions come from `step_contexts()`, which reads `metadata.prompt_tokens` out of the graph JSON: step *i*'s feature sits at position `ntok_i - 1`, read directly off graph metadata rather than recomputed. Two independent guards check this: one asserts each derived position matches the node's own recorded `ctx_idx`; the other raises if a position falls outside the shorter measurement sequence, which is what catches an execution feature reaching the control pool.

### Load & filter graph files

Reads every `step-NN-*.json` file per prompt slug. Each node's `influence` field in the JSON is a cumulative share (nodes sorted by real influence descending, running total divided by the total), not a per-node magnitude -- the most influential node gets the smallest value. Differencing adjacent values in ascending order recovers each node's own share first; only then are transcoder nodes kept above the influence threshold and recorded as `(layer, feature, influence)`.

### Normalize & build feature timeline

Divides each feature's influence by its step's total to produce a normalized share, plus a within-step percentile rank. Indexed as `feature_timeline[(layer, feat)][step]`.

### Classify planning vs execution features

For every `(layer, feature)` pair, computes `first_step`, `peak_step`/`peak_val`, `rhyme_val` (normalized influence at the rhyme step), and `sustain_ratio = rhyme_val / peak_val`. Features peaking **before** the rhyme step are planning features; those peaking **at or after** are execution features and excluded from measured populations, though still reported under counts.

Features in the last layer are excluded outright (`feature_stats(n_layers=...)`) — see Glossary.

### Populations

`analyze_prompt()` builds four populations, then `build_measurement_set()` unions them, keyed `(layer, feat, position)`:

- **candidate** — shipped influence/percentile/sustain-ratio cutoffs
- **superset** — the loosest cell of the threshold-sensitivity grid
- **near_miss** — just below the candidate cutoffs
- **random_control** — a matched random sample

There is no measurement cap on any of these — `POPULATION_CEILING` is a tripwire that raises rather than truncates, not a parameter to tune.

### Causal interventions

For each measured feature, suppress its activation to 0 at its recorded position and measure `logit_drop` / `prob_drop` on the rhyme token, plus rank shift. Both the feature's `first_step` and `peak_step` context are measured and reported (a feature can yield up to two rows; both analysis scripts collapse with `best_per_feature()` before any per-feature statistic).

### Output

Each prompt/size writes `experiment/tracing/circuit_tracing_results_{size}_{slug}.json` containing `config` (thresholds, band cutoffs, `excludes_last_layer`, `n_layers`, `code_version`), `statistics`, `candidates`, `early_spikes`, and `downstream_effects` (suppression results, keyed by population and step).

**Pooling markers.** `config.selection_percentile_population` and `config.code_version` identify which methodology generation a result file belongs to. `comparing.py` checks these markers agree across every file in a pool before running any statistic, and raises if they don't — do not narrow `--sizes` to silence it, that just drops a size from the regression instead of fixing the mismatch.

---

## Step 4 — Threshold Sensitivity

**File:** `threshold_sensitivity.py`

GPU-free — a pure join against `tracing.py`'s already-measured populations, so it doesn't load a model itself. Sweeps `INFLUENCE_GRID` / `PERCENTILE_GRID` / `SUSTAIN_GRID` (defined in `tracing.py`, since the loosest cell is what sizes the superset that has to be measured upstream) and reports, per grid cell, one of `hit` / `measured_at_other_position` / `not_measured` / `measurement_failed` — only the last is a defect; the first three are distinguishable causes of sweep coverage.

Writes `experiment/threshold/threshold_sensitivity_{size}_{slug}.json`.

---

## Step 5 — Cross-Model Comparison

**File:** `comparing.py`

Aggregates the per-prompt, per-size results, split by rhyme label (`rhyme` / `near_rhyme` / `repetition` / `none` / etc.), and regresses planning measures — e.g. mean `logit_drop` over the top-10 candidates — on `log10(rhymes_all)` using `scipy.stats`. Checks pooling markers (see Step 3) before running any statistic.

Prints a report to stdout and writes `experiment/comparison.json`. `RESULTS_DIR` must stay in sync with `tracing.py`'s.

---

## Other scripts

**`chat_prefix_frame_check.py`** — a standalone diagnostic, not part of the five-stage pipeline. Checks whether the chat-prefix frame used during attribution changes what the model actually generates, relative to the bare prompt used during generation.

**`run_generation_all.sh` / `run_tracing_all.sh` / `run_threshold_all.sh`** — shell runners that loop each stage over all three sizes.

**`experiment/tests/`** — regression tests for the GPU-free half of `tracing.py` and for `downstream_effects_addon.py`'s measurement layer (`python -m pytest experiment/tests`). Covers position derivation (`step_contexts`, `ctx_idx` agreement, out-of-bounds detection), last-layer exclusion, population tagging/dedup, pooling-marker enforcement, and the `logsumexp`/`prob`/`rank` consistency of suppression measurements. Not in CI, but runs in milliseconds.

---

## Environment setup for `experiment/`

Anything importing `tools/prompt_set.py` needs more than `pip install -e .`, because NLTK corpora are runtime downloads rather than package dependencies:

```bash
pip install -e .                                            # incl. cmudict, pronouncing, scipy
python -m nltk.downloader averaged_perceptron_tagger_eng wordnet
```

`averaged_perceptron_tagger_eng` is **required** — `is_content_word` calls `nltk.pos_tag`, and the last-content-word rule depends on it. `wordnet` is optional but wanted — see Step 2.
