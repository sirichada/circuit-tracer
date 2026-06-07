# Investigating Forward Planning Behaviors in Smaller LLMs through Transcoder Circuit Tracing 

## Overview

This project investigates whether transformer language models engage in **advance planning** when generating rhyming couplets. Specifically: when a model generates the second line of a couplet, does it "decide" on the rhyme word before it starts writing, or does it figure it out token by token?

The pipeline runs across three model sizes — Gemma-3 270M, 1B, and 4B — using [Gemma-Scope-2](https://huggingface.co/google/gemma-scope-2) transcoders to replace MLP layers with interpretable sparse features. Attribution graphs are built for every generated token, feature timelines are extracted and analyzed, and causal interventions (feature suppression) are used to verify which features are causally necessary for the rhyme.

Code documentation was generated with Claude (Anthropic).

---

## Pipeline Overview

```
generation-gemma-3-{270M,1B,4b}.ipynb
        │  generates token-by-token, runs attribution per step
        │  outputs: graph_files/gemma-3-{size}/step-NN-{token}.json
        ▼
tracing-{270m,1b,4b}.py
        │  loads graph files, extracts feature timelines, classifies
        │  planning vs execution features, runs causal interventions
        │  outputs: circuit_tracing_results_{size}.json
        ▼
comparing.py
        │  loads all three JSON results, compares feature overlap,
        │  scaling behaviour, and suppression effects across model sizes
        ▼
        terminal output (printed insights 1–8 + summary table)
```

---

## Step 1 — Graph Generation

**Files:** `generation-gemma-3-270M.ipynb`, `generation-gemma-3-1B.ipynb`, `generation-gemma-3-4b.ipynb`

### What they do

Each notebook loads a Gemma-3 model patched with Gemma-Scope-2 transcoders, then runs two loops back-to-back:

**Generation loop** — greedy-decodes the second line of the prompt one token at a time, stopping at punctuation or EOS. Every generated token is recorded in `token_trace` with its token ID, string, probability, and top-10 alternatives.

**Attribution loop** — for every token in `token_trace`, re-runs the model with `circuit_tracer.attribute()` to produce a full attribution graph: which transcoder features at which layer and position were most responsible for predicting that token. The graph is saved as a JSON file named `step-NN-{token}.json`.

### Prompt

```
A rhyming couplet:
He saw a carrot and had to grab it,
```

The model completes the second line. For the 4B model this produces *"He ate it and then he had to crap it."* — the rhyme word is `crap` at step 8. The 270M and 1B models rhyme on `it` at step 19.

### Key parameters (cell-8)

| Parameter | Value | Effect |
|---|---|---|
| `max_n_logits` | 5 | Maximum number of output logits to attribute from |
| `desired_logit_prob` | 0.95 | Stops attributing once this probability mass is covered |
| `max_feature_nodes` | 1028 | Maximum transcoder features included in the graph |
| `batch_size` | 8 | Batch size during attribution |
| `offload` | `"disk"` | Offloads intermediate tensors to disk to save GPU memory |

### Outputs

Graph files are written to `./graph_files/gemma-3-{size}/`. Each file contains a list of nodes (transcoder features, residual stream positions, logit targets) and edges (influence weights between them). A `graph-metadata.json` manifest is also written to enable the circuit-tracer viewer.

### Model/transcoder details

| Notebook | Model | Transcoder repo | Layers |
|---|---|---|---|
| 270M | `google/gemma-3-270m` | `gemma-scope-2-270m-pt` | 18 |
| 1B | `google/gemma-3-1b-pt` | `gemma-scope-2-1b-pt` | 26 |
| 4B | `google/gemma-3-4b-pt` | `gemma-scope-2-4b-pt` | 34 |

All transcoders use width `16k`, L0 `small`, and hook points `hook_resid_mid` (input) → `hook_mlp_out` (output).

---

## Step 2 — Feature Tracing & Causal Intervention

**Files:** `tracing-270m.py`, `tracing-1b.py`, `tracing-4b.py`

Each script is structurally identical; differences are the graph directory, model name, rhyme token, and rhyme step.

| Script | Rhyme token | Rhyme step |
|---|---|---|
| `tracing-270m.py` | `" it"` | 19 |
| `tracing-1b.py` | `" it"` | 19 |
| `tracing-4b.py` | `" crap"` | 8 |

### Sections

#### Load & filter graph files
Reads every `step-NN-*.json` file from the graph directory. For each step, keeps only transcoder nodes with `|influence| ≥ 0.001` and records `(layer, feature, influence)`. Stores raw rows per step in `step_features_raw` and the sum of absolute influence per step in `step_total_influence`.

#### Normalize & build feature timeline
Divides each feature's influence by its step's total to produce a normalized share (so features are comparable across steps). Also computes a within-step percentile rank (top feature = 100th percentile). Results are indexed as `feature_timeline[(layer, feat)][step]` and `feature_percentiles[(layer, feat)][step]`.

#### Classify planning vs execution features
For every unique `(layer, feature)` pair seen across all steps, computes:
- `first_step` — earliest step the feature was active
- `peak_step` / `peak_val` — when and how strongly it peaked
- `rhyme_val` — its normalized influence at the rhyme step
- `sustain_ratio = rhyme_val / peak_val` — how much of its peak influence it retains by rhyme time

Features peaking **before** the rhyme step go into `planning_features`; those peaking **at or after** go into `execution_features`.

#### Identify rhyme-circuit candidates
Filters `planning_features` to those that are also active at the rhyme step, with:
- rhyme-step percentile ≥ 50 (still prominent at rhyme time)
- `sustain_ratio` ≥ 0.3 (retains at least 30% of peak influence)

These are the features that activate early and stay relevant — the strongest candidates for being part of the rhyme-planning circuit.

#### Per-step top features (qualitative view)
Prints the top 5 features by normalized influence for every step, with percentile rank. The rhyme step is annotated. Used as a sanity check.

#### Early spike detection
Finds features that first hit ≥ 70th percentile very early in the sequence, sorted by how early. Captures features that become prominent before they'd need to be for simple next-token prediction.

#### Temporal clustering
Splits all features into three temporal bands by peak step:
- **EARLY** — steps 0 to `RHYME_STEP // 3`
- **MID** — steps `RHYME_STEP // 3` to `2 * RHYME_STEP // 3`
- **LATE** — steps `2 * RHYME_STEP // 3` onward

Reports feature counts, most active layers, and top features per band.

#### Causal interventions
The core experiment. For each rhyme-circuit candidate, the feature's activation is zeroed out (suppressed to 0) at two different points:
- **Peak-step suppression** — suppress at the step where the feature is strongest
- **First-step suppression** — suppress at the feature's very first active step

For each suppression, measures the drop in the model's probability of generating the rhyme token, and the shift in that token's rank. The top 10 features by probability drop are re-run with full generation to show the before/after output text. The two suppression strategies are compared side by side for shared features.

#### Pseudo-CLERP
For each top candidate and early-spike feature, multiplies the transcoder's decoder weight vector `W_dec[feat]` by the unembedding matrix `W_U` to get a vocabulary-space projection — a quick proxy for "what tokens does this feature causally predict?" Run for the top rhyme-circuit candidates and the top early-spike features.

### Output

Each script writes `circuit_tracing_results_{size}.json` containing:
- `config` — graph directory, rhyme step, thresholds, band cutoffs
- `statistics` — counts of unique features, planning/execution features, candidates, early spikes
- `candidates` — full list of rhyme-circuit candidates with their stats
- `early_spikes` — full list of early-spike features
- `downstream_effects` — peak-step and first-step suppression results (top 30 by probability drop for each strategy)

---

## Step 3 — Cross-Model Comparison

**File:** `comparing.py`

Loads `circuit_tracing_results_{270m,1b,4b}.json` and runs eight comparative analyses, printed to the terminal.

| Insight | What it shows |
|---|---|
| 1 — Feature persistence | How many rhyme-circuit candidates are shared across model sizes (all three, pairs, unique to one) |
| 2 — Peak step timing | Distribution of candidate peak steps per model — does planning happen earlier or later at larger scale? |
| 3 — Sustain ratio stats | Mean/min/max sustain ratio per model — do larger models maintain features more persistently? |
| 4 — Percentile consistency | Top-10 candidates per model ranked by peak influence, with their rhyme-step percentile |
| 5 — Planning vs execution ratio | What fraction of all features peak before vs at/after the rhyme step, at each scale |
| 6 — Temporal band distribution | How candidates distribute across EARLY/MID/LATE bands per model |
| 7 — Candidate growth & early spikes | How candidate counts and early-spike prevalence scale with model size |
| 8 — Suppression effects | Max/average probability drop when suppressing candidates, and the single most impactful feature per model |

A summary table at the end prints unique features, candidates, early spikes, and planning/execution percentages for all three models side by side.
