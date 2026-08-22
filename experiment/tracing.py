"""Shared tracing pipeline for all Gemma-3 sizes.

Consolidates what used to be ~730 near-identical lines in each of
`tracing-{270m,1b,4b}.py`. Those three now supply only their config and call
`run()` here. Keeping one copy is what stops `RHYME_TOKEN` / `GRAPH_DIR` /
the checkpoint names from going stale in three places at once.

Two halves:

  * `analyze_prompt()` and everything it calls is **pure** -- graph JSON in,
    statistics out. No model, no GPU. Run it anywhere.
  * `run_interventions()` needs the model on a GPU.

**All GPU work in the experiment lives here.** `threshold_sensitivity.py` used
to load a model of its own to measure near-misses and the random control; it no
longer does. This module measures the union of every population any downstream
analysis needs -- shipped candidates, the loosest-grid-cell superset,
near-misses, and the matched-random control -- in one model load per size, tags
each row with the populations it belongs to, and lets the analysis side re-slice
without re-measuring.

Positions, not steps
--------------------
The second element of an intervention tuple is a **token index into the
tokenized input**, not a generation step. Suppression positions come from
`step_contexts()`, which reads `metadata.prompt_tokens` out of the graph JSON:
step *i*'s feature sits at position `ntok_i - 1`. This was measured across
4B/1B/270M x {realm, ten, original}: `ntok` is exactly linear in the step, each
step's tokens are a strict prefix-extension of step 0's, and 99.0-100% of
transcoder nodes sit at `ctx_idx == ntok - 1`. Every derived position is checked
against the node's own recorded `ctx_idx`.

Per-prompt rhyme targets come from `rhyme_labels.json` (produced by
`rhyme_labels.py`), never from hardcoded constants.

    python experiment/tracing-1b.py                    # analysis + interventions
    python experiment/tracing-1b.py --no-interventions # GPU-free half only
    python experiment/tracing-1b.py --slugs realm ten  # selected prompts
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPERIMENT = Path(__file__).parent
REPO = EXPERIMENT.parent
PROMPT_SET_PATH = REPO / "tools" / "prompt_set.json"
LABELS_PATH = EXPERIMENT / "rhyme_labels.json"
# Per-prompt results live in their own directory rather than loose in
# experiment/. Note .gitignore:90 already ignores `experiment/tracing`, so
# these outputs are untracked by default -- intentional, they are regenerable.
RESULTS_DIR = EXPERIMENT / "tracing"

# --- analysis constants (identical across sizes; were duplicated per script) ---
INFLUENCE_THRESHOLD = 0.001
CANDIDATE_MIN_RHYME_PERCENTILE = 50.0
CANDIDATE_MIN_SUSTAIN = 0.3
EARLY_SPIKE_PERCENTILE = 70.0  # descriptive only -- undefended and unswept
TOP_N_REPORTED = 15

# --- sweep grids -------------------------------------------------------------
# Defined here rather than in threshold_sensitivity.py because the *loosest* cell
# determines the superset this module has to measure: sizing the measurement set
# against one grid while sweeping another is how coverage gaps appear.
#
# Each grid straddles the shipped value. The influence and percentile grids used
# to be skewed strict (one point below the shipped value, two or three above),
# which explored the opposite direction from the reviewer concern -- that the
# cutoffs *discard* real signal. Both are now symmetric.
INFLUENCE_GRID = (0.00025, 0.0005, 0.001, 0.002, 0.004)
PERCENTILE_GRID = (30.0, 40.0, 50.0, 60.0, 70.0)
SUSTAIN_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)

# --- near-miss margins: how far below each cutoff still counts as "just missed" ---
SUSTAIN_MARGIN = 0.15  # catches sustain_ratio in [0.15, 0.30)
PERCENTILE_MARGIN = 15.0  # catches rhyme_percentile in [35, 50)

RANDOM_SEED = 0

# There is no measurement cap. `MAX_CANDIDATES_MEASURED = 50` used to be applied
# as a silent slice, and shipped-cell populations run 19-192 (median ~62) -- so
# every previously reported mean was a mean over a truncated, high-influence
# prefix, which is precisely the "generically high-influence" confound the
# reviewers raised. The population is bounded by construction (a candidate must
# be present at the rhyme step, so it cannot exceed one step's node count), and
# the widened loosest cell is only 1.4-2.8x the shipped set. This ceiling exists
# so a future change that makes the population explode fails visibly instead of
# quietly truncating -- it is a tripwire, not a parameter.
POPULATION_CEILING = 1000

# Generation is ~20 sequential forwards and dominates runtime, so only a subset
# of measured features gets a saved continuation. Ranked by `logit_drop`: Zhang &
# Nanda recommend against probability, which cannot register an effect once the
# target probability is near zero. Ranked *within* each step-key condition
# separately, then unioned -- a pooled ranking would let
# whichever condition has systematically larger effects crowd the other out.
GENERATION_TOP_N = 20

# Both suppression points are measured and both are reported. Which is primary is
# a writing-time decision deliberately left open; the pre-commitment that matters
# is reporting both. `first_step` is defined against INFLUENCE_THRESHOLD and so
# drifts as the sweep varies it, while `peak_step` is an argmax over normalised
# influence -- the sweep measures that drift directly, so the eventual
# designation can be argued from data rather than from the definitions.
STEP_KEYS = ("peak_step", "first_step")

WIDTH = "16k"
L0 = "small"

STEP_RE = re.compile(r"step-(\d+)-(.*)\.json$")


@dataclass(frozen=True)
class SizeConfig:
    size: str
    model_name: str
    transcoder_repo: str
    n_layers: int

    @property
    def graph_dir(self) -> Path:
        return EXPERIMENT / "graphs" / f"gemma-3-{self.size}-it"

    def results_path(self, slug: str) -> Path:
        return RESULTS_DIR / f"circuit_tracing_results_{self.size}_{slug}.json"


CONFIGS = {
    "270m": SizeConfig("270m", "google/gemma-3-270m-it", "google/gemma-scope-2-270m-it", 18),
    "1b": SizeConfig("1b", "google/gemma-3-1b-it", "google/gemma-scope-2-1b-it", 26),
    "4b": SizeConfig("4b", "google/gemma-3-4b-it", "google/gemma-scope-2-4b-it", 34),
}


# ---------------------------------------------------------------- pure analysis


def parse_node_ids(node: dict) -> tuple[int | None, int | None]:
    """(layer, local_feat) from a node dict.

    Only `jsNodeId` is trusted. The obvious-looking fallback to `node["feature"]`
    is **wrong**: that field is `cantor_pairing(layer, feat_idx)`
    (`frontend/graph_models.py:52`), not a feature index, so reading it would
    hand back plausible-looking garbage. Raise instead -- a graph without
    `jsNodeId` means the export format changed and every downstream feature
    identity is suspect.
    """
    m = re.match(r"^(\d+)_(\d+)-", node.get("jsNodeId", ""))
    if m:
        return int(m.group(1)), int(m.group(2))
    if node.get("feature") is not None:
        raise ValueError(
            f"node has no parseable jsNodeId (got {node.get('jsNodeId')!r}) but does carry "
            "`feature`, which is a cantor pairing of (layer, feat) and must not be read as a "
            "feature index. The graph export format has changed."
        )
    return None, None


def step_contexts(slug_dir: Path) -> dict[int, dict]:
    """Per step: the exact text and token count that was attributed.

    `metadata.prompt` is `CHAT_PREFIX + prompt_text + generated_tokens[:i]` --
    the literal string `attribute()` ran on -- and `metadata.prompt_tokens` is
    its tokenization. Suppression position for step *i* is `ntok_i - 1`.

    Nothing here does string arithmetic on CHAT_PREFIX or re-tokenizes anything;
    both would reintroduce the frame and special-token divergences that put the
    original interventions on the wrong position.
    """
    contexts: dict[int, dict] = {}
    for path in sorted(slug_dir.glob("step-*.json")):
        m = STEP_RE.search(path.name)
        if not m:
            continue
        meta = json.loads(path.read_text()).get("metadata", {})
        prompt = meta.get("prompt")
        tokens = meta.get("prompt_tokens")
        if prompt is None or not tokens:
            raise ValueError(f"{path.name}: metadata lacks prompt/prompt_tokens")
        contexts[int(m.group(1))] = {
            "prompt": prompt,
            "ntok": len(tokens),
            "position": len(tokens) - 1,
            "token": m.group(2).replace("_", " "),
        }
    return contexts


def load_raw_step_nodes(
    slug_dir: Path, floor: float
) -> tuple[dict[int, list[dict]], dict[int, str]]:
    """Every transcoder node in every step above `floor`, with its `ctx_idx`.

    Read once at the loosest floor any caller will use, then filtered in memory
    by `filter_at`. Re-globbing per grid point would parse the same JSON five
    times over.

    `ctx_idx` is the token position the node was attributed at. It is carried
    through (it used to be dropped here) because it is the independent check on
    every derived suppression position.
    """
    step_nodes: dict[int, list[dict]] = {}
    step_tokens: dict[int, str] = {}

    for path in sorted(slug_dir.glob("step-*.json")):
        m = STEP_RE.search(path.name)
        if not m:
            continue
        data = json.loads(path.read_text())

        rows = []
        for node in data.get("nodes", []):
            if "transcoder" not in node.get("feature_type", ""):
                continue
            inf = node.get("influence") or 0
            if inf == 0 or abs(inf) < floor:
                continue
            layer, feat = parse_node_ids(node)
            if layer is None:
                continue
            rows.append(
                {
                    "layer": layer,
                    "feat": feat,
                    "influence": abs(inf),
                    "raw_inf": inf,
                    "ctx_idx": node.get("ctx_idx"),
                }
            )

        step_nodes[int(m.group(1))] = rows
        step_tokens[int(m.group(1))] = m.group(2).replace("_", " ")

    return step_nodes, step_tokens


def filter_at(step_nodes: dict[int, list[dict]], threshold: float) -> tuple[dict, dict]:
    """Apply an influence threshold to pre-read nodes; return (rows, totals)."""
    step_features, step_total = {}, {}
    for step_idx, rows in step_nodes.items():
        kept = [r for r in rows if r["influence"] >= threshold]
        step_features[step_idx] = kept
        step_total[step_idx] = sum(r["influence"] for r in kept)
    return step_features, step_total


def load_step_features(slug_dir: Path) -> tuple[dict, dict, dict]:
    """Shipped-threshold view. Thin wrapper kept for callers and tests."""
    step_nodes, step_tokens = load_raw_step_nodes(slug_dir, INFLUENCE_THRESHOLD)
    step_features, step_total = filter_at(step_nodes, INFLUENCE_THRESHOLD)
    return step_features, step_total, step_tokens


def build_timeline(step_features: dict, step_total: dict) -> tuple[dict, dict]:
    """Normalize each feature's influence by its step's total, and rank it
    within the step. Both are *within-step* measures, which is what makes
    values from separately-computed graphs comparable at all.
    """
    timeline: dict[tuple, dict[int, float]] = defaultdict(dict)
    percentiles: dict[tuple, dict[int, float]] = defaultdict(dict)

    for step_idx in sorted(step_features):
        rows, total = step_features[step_idx], step_total[step_idx]
        if total == 0:
            continue
        ordered = sorted(rows, key=lambda r: -r["influence"])
        for rank, row in enumerate(ordered):
            key = (row["layer"], row["feat"])
            timeline[key][step_idx] = row["influence"] / total
            percentiles[key][step_idx] = 100.0 * (len(ordered) - rank) / len(ordered)

    return dict(timeline), dict(percentiles)


def collect_ctx_idx(step_features: dict) -> dict[tuple, dict[int, int]]:
    """(layer, feat) -> {step: ctx_idx}, for checking derived positions."""
    out: dict[tuple, dict[int, int]] = defaultdict(dict)
    for step_idx, rows in step_features.items():
        for row in rows:
            if row.get("ctx_idx") is not None:
                out[(row["layer"], row["feat"])][step_idx] = int(row["ctx_idx"])
    return dict(out)


def feature_stats(timeline: dict, percentiles: dict, rhyme_step: int, n_layers: int) -> list[dict]:
    """Per-feature timing/influence summaries, over the *measurable* features only.

    Final-layer features are dropped: attention precedes the MLP, so nothing
    crosses positions after `blocks.{n_layers-1}.hook_mlp_out` and suppressing
    one before the readout is bit-identical to baseline by construction.

    `n_layers` is required rather than defaulted so a new call site cannot
    silently re-admit them.

    Two percentiles are returned. `rhyme_percentile`/`peak_percentile` rank
    among all graph nodes and are what gets reported;
    `*_measurable` rank among the features that survive the layer filter and are
    what selection tests. They must not be the same number: dropped features
    outrank survivors, so ranking survivors against a distribution containing
    them pushes them below `CANDIDATE_MIN_RHYME_PERCENTILE`. Normalization in
    `build_timeline`/`filter_at` is untouched, so the reported denominators still
    describe the whole graph.
    """
    # Ranking by normalized influence is identical to ranking by raw influence
    # (the per-step divisor is constant), so the measurable percentiles can be
    # rebuilt from `timeline` alone.
    survivors = {k: v for k, v in timeline.items() if v and k[0] < n_layers - 1}

    by_step: dict[int, list[tuple[float, tuple]]] = defaultdict(list)
    for key, step_inf in survivors.items():
        for step_idx, val in step_inf.items():
            by_step[step_idx].append((val, key))

    # Same rank formula as `build_timeline`, over the smaller population.
    measurable: dict[tuple, dict[int, float]] = defaultdict(dict)
    for step_idx, entries in by_step.items():
        entries.sort(key=lambda t: -t[0])
        n = len(entries)
        for rank, (_, key) in enumerate(entries):
            measurable[key][step_idx] = 100.0 * (n - rank) / n

    stats = []
    for key, step_inf in survivors.items():
        steps_sorted = sorted(step_inf)
        peak_val = max(step_inf.values())
        peak_step = max(step_inf, key=lambda s: step_inf[s])
        rhyme_val = step_inf.get(rhyme_step, 0.0)
        stats.append(
            {
                "feat_key": key,
                "first_step": steps_sorted[0],
                "last_step": steps_sorted[-1],
                "n_steps": len(steps_sorted),
                "peak_step": peak_step,
                "peak_val": peak_val,
                "peak_percentile": percentiles[key].get(peak_step, 0.0),
                "peak_percentile_measurable": measurable[key].get(peak_step, 0.0),
                "rhyme_val": rhyme_val,
                "rhyme_percentile": percentiles[key].get(rhyme_step, 0.0),
                "rhyme_percentile_measurable": measurable[key].get(rhyme_step, 0.0),
                "sustain_ratio": rhyme_val / peak_val if peak_val > 0 else 0.0,
            }
        )
    return stats


def candidate_keys(
    stats: list[dict], rhyme_step_features: set, rhyme_step: int, min_pct: float, min_sustain: float
) -> list[dict]:
    """The candidate predicate with both cutoffs as arguments.

    Planning features (peaking strictly before the rhyme) that are also prominent
    *at* the rhyme step and hold a good share of their peak influence until then.

    Prominence is tested against `rhyme_percentile_measurable` -- rank among the
    features that survive the layer filter -- not the all-nodes `rhyme_percentile`
    that gets reported. See `feature_stats` for why the two must not be the same
    number here.
    """
    out = [
        s
        for s in stats
        if s["peak_step"] < rhyme_step
        and s["feat_key"] in rhyme_step_features
        and s["rhyme_percentile_measurable"] >= min_pct
        and s["sustain_ratio"] >= min_sustain
    ]
    out.sort(key=lambda x: -(x["peak_val"] + x["rhyme_val"]))
    return out


def select_candidates(planning: list[dict], rhyme_step_features: set) -> list[dict]:
    """Shipped-cutoff candidates, drawn from an already-filtered planning list."""
    out = [
        e
        for e in planning
        if e["feat_key"] in rhyme_step_features
        and e["rhyme_percentile_measurable"] >= CANDIDATE_MIN_RHYME_PERCENTILE
        and e["sustain_ratio"] >= CANDIDATE_MIN_SUSTAIN
    ]
    out.sort(key=lambda x: -(x["peak_val"] + x["rhyme_val"]))
    return out


def split_populations(stats: list[dict], rhyme_step_features: set, rhyme_step: int) -> dict:
    """Partition rhyme-step features into candidates / near-misses / the rest.

    The three groups are disjoint by construction, so the random control can never
    accidentally draw a feature the candidate filter nearly kept.

    **Execution features are excluded from all three.** They peak at or after the
    rhyme step, so their suppression position `ntok_peak - 1` can fall past the end
    of the measurement sequence, which spans only through the rhyme step. That is a
    hard IndexError, and putting them in the control pool would have made the
    control quietly smaller than the candidate set. Excluding them is also the
    better control: suppressing a feature at a position *after* the rhyme cannot
    speak to whether it caused the rhyme, so matching on timing as well as on
    influence is what the comparison actually needs. They stay in the reported
    population counts under `execution`.
    """
    at_rhyme = [s for s in stats if s["feat_key"] in rhyme_step_features]

    candidates, near_misses, rest, execution = [], [], [], []
    for s in at_rhyme:
        if s["peak_step"] >= rhyme_step:
            execution.append(s)
            continue
        # Measurable-population percentile, matching `select_candidates`. Using
        # the all-nodes one here would make near-miss membership disagree with
        # candidate membership, and the two pools have to partition cleanly.
        passed_pct = s["rhyme_percentile_measurable"] >= CANDIDATE_MIN_RHYME_PERCENTILE
        passed_sus = s["sustain_ratio"] >= CANDIDATE_MIN_SUSTAIN
        if passed_pct and passed_sus:
            candidates.append(s)
            continue
        near_pct = (
            (CANDIDATE_MIN_RHYME_PERCENTILE - PERCENTILE_MARGIN)
            <= s["rhyme_percentile_measurable"]
            < CANDIDATE_MIN_RHYME_PERCENTILE
        )
        near_sus = (
            (CANDIDATE_MIN_SUSTAIN - SUSTAIN_MARGIN) <= s["sustain_ratio"] < CANDIDATE_MIN_SUSTAIN
        )
        # Within the margin on whichever filter it failed, and passing the other.
        if (
            (not passed_pct and passed_sus and near_pct)
            or (not passed_sus and passed_pct and near_sus)
            or (not passed_pct and not passed_sus and near_pct and near_sus)
        ):
            near_misses.append(s)
        else:
            rest.append(s)

    candidates.sort(key=lambda x: -(x["peak_val"] + x["rhyme_val"]))
    near_misses.sort(key=lambda x: -(x["sustain_ratio"] + x["rhyme_percentile"] / 100))
    return {
        "candidates": candidates,
        "near_misses": near_misses,
        "rest": rest,
        "execution": execution,
    }


def sample_matched_control(candidates: list[dict], rest: list[dict], seed: int) -> list[dict]:
    """Draw non-candidates matched to the candidate set's `rhyme_val` distribution.

    Sampling uniformly from `rest` would stack the comparison: most non-candidates sit
    near zero influence, so they'd fail to move anything for reasons that have nothing
    to do with the selection filter. Matching on influence decile makes the control
    answer the question actually being asked.

    `rest` is already restricted to `peak_step < rhyme_step` by `split_populations`,
    so every drawn feature has an in-range suppression position.
    """
    if not candidates or not rest:
        return []

    edges = sorted(c["rhyme_val"] for c in candidates)
    n_bins = 10

    def bin_of(val: float) -> int:
        lo, hi = edges[0], edges[-1]
        if hi <= lo:
            return 0
        return min(n_bins - 1, int(n_bins * (val - lo) / (hi - lo))) if lo <= val <= hi else -1

    pool: dict[int, list[dict]] = defaultdict(list)
    for s in rest:
        pool[bin_of(s["rhyme_val"])].append(s)

    wanted: dict[int, int] = defaultdict(int)
    for c in candidates:
        wanted[bin_of(c["rhyme_val"])] += 1

    rng = random.Random(seed)
    drawn: list[dict] = []
    shortfall = 0
    for b, n in sorted(wanted.items()):
        available = pool.get(b, [])
        take = min(n, len(available))
        drawn.extend(rng.sample(available, take))
        shortfall += n - take

    # Backfill from the nearest populated bins so the control isn't silently smaller
    # than the candidate set, which would make the two means incomparable.
    if shortfall:
        used = {id(s) for s in drawn}
        spare = [s for s in rest if id(s) not in used]
        spare.sort(key=lambda s: -s["rhyme_val"])
        drawn.extend(spare[:shortfall])

    return drawn


def find_early_spikes(percentiles: dict, stats_by_key: dict) -> list[dict]:
    """Descriptive only: EARLY_SPIKE_PERCENTILE is neither defended nor swept."""
    spikes = []
    for key, step_pct in percentiles.items():
        spike_step = next(
            (s for s in sorted(step_pct) if step_pct[s] >= EARLY_SPIKE_PERCENTILE), None
        )
        if spike_step is None or key not in stats_by_key:
            continue
        st = stats_by_key[key]
        spikes.append(
            {
                "feat_key": key,
                "early_spike_step": spike_step,
                "peak_step": st["peak_step"],
                "peak_val": st["peak_val"],
                "rhyme_val": st["rhyme_val"],
                "rhyme_percentile": st["rhyme_percentile"],
                "sustain_ratio": st["sustain_ratio"],
            }
        )
    spikes.sort(key=lambda x: x["early_spike_step"])
    return spikes


def temporal_bands(timeline: dict, rhyme_step: int) -> dict[str, list]:
    """Descriptive only: the rhyme_step//3 tertile bands are an arbitrary split."""
    early_cutoff, mid_cutoff = rhyme_step // 3, (2 * rhyme_step) // 3
    bands: dict[str, list] = {"EARLY (planning)": [], "MID (buildup)": [], "LATE (execution)": []}
    for key, step_inf in timeline.items():
        if not step_inf:
            continue
        peak_step = max(step_inf, key=lambda s: step_inf[s])
        if peak_step <= early_cutoff:
            bands["EARLY (planning)"].append(key)
        elif peak_step <= mid_cutoff:
            bands["MID (buildup)"].append(key)
        else:
            bands["LATE (execution)"].append(key)
    return bands


def _serialize(entries: list[dict], extra: tuple[str, ...] = ()) -> list[dict]:
    keys = (
        "first_step",
        "peak_step",
        "peak_val",
        "rhyme_val",
        "sustain_ratio",
        "rhyme_percentile",
        # Both are emitted: the all-nodes one is descriptive, the measurable one
        # is what selection actually tested. Reporting only the first would make
        # a shipped candidate look like it failed its own cutoff.
        "rhyme_percentile_measurable",
    ) + extra
    out = []
    for e in entries:
        row = {"layer": e["feat_key"][0], "feat": e["feat_key"][1]}
        for k in keys:
            if k in e:
                row[k] = float(e[k]) if isinstance(e[k], float) else e[k]
        out.append(row)
    return out


# --------------------------------------------------------- measurement set build


def build_measurement_set(
    populations: dict[str, list[dict]],
    contexts: dict[int, dict],
    ctx_idx_by_key: dict[tuple, dict[int, int]],
    rhyme_step: int,
    n_layers: int,
) -> tuple[list[dict], dict]:
    """One deduplicated row per (layer, feat, position) across every population.

    Roughly half of all candidates have `first_step == peak_step` (50.8%, measured
    over 4B/1B/270M x 4 slugs, n=2560), so looping the two step keys blindly meant
    measuring the identical intervention at the identical position twice. Rows are
    keyed by position and tagged with every `(population, step_key)` that produced
    them, which covers both conditions over the full population at ~75% of the
    forward passes.

    Returns (rows, diagnostics). Diagnostics record ctx_idx agreement: the derived
    position is `ntok_step - 1`, and the node's own `ctx_idx` is an independent
    witness to it. 0-4 nodes per graph legitimately sit elsewhere, so a handful of
    mismatches is expected and logged rather than fatal.
    """
    rhyme_ntok = contexts[rhyme_step]["ntok"]
    rows: dict[tuple[int, int, int], dict] = {}
    checked = matched = 0
    mismatches: list[dict] = []

    for pop_name, entries in populations.items():
        for e in entries:
            layer, feat = e["feat_key"]
            for step_key in STEP_KEYS:
                step = e[step_key]
                if step not in contexts:
                    continue
                position = contexts[step]["position"]

                recorded = ctx_idx_by_key.get((layer, feat), {}).get(step)
                if recorded is not None:
                    checked += 1
                    if recorded == position:
                        matched += 1
                    elif len(mismatches) < 20:
                        mismatches.append(
                            {
                                "layer": layer,
                                "feat": feat,
                                "step": step,
                                "derived_position": position,
                                "ctx_idx": recorded,
                            }
                        )

                key = (layer, feat, position)
                row = rows.get(key)
                if row is None:
                    row = rows[key] = {
                        "layer": layer,
                        "feat": feat,
                        "position": position,
                        "step": step,
                        "peak_step": e["peak_step"],
                        "first_step": e["first_step"],
                        "populations": [],
                        "step_keys": [],
                    }
                if pop_name not in row["populations"]:
                    row["populations"].append(pop_name)
                if step_key not in row["step_keys"]:
                    row["step_keys"].append(step_key)

    ordered = sorted(rows.values(), key=lambda r: (r["position"], r["layer"], r["feat"]))
    for r in ordered:
        # Redundant with measure_features' own guard, but this one names the step.
        assert r["position"] < rhyme_ntok, (
            f"L{r['layer']} F{r['feat']} step {r['step']} -> position {r['position']} "
            f">= measurement length {rhyme_ntok}"
        )
        # Defence in depth, and a *different* check from the two above: those catch
        # a position defect, this catches a population built from a stats list that
        # never went through `feature_stats`' layer filter. Such a row would measure
        # a guaranteed zero and pool it as a null.
        if r["layer"] >= n_layers - 1:
            raise RuntimeError(
                f"L{r['layer']} F{r['feat']} is in the final layer (n_layers={n_layers}) "
                f"at position {r['position']}, which has no causal path to the readout. "
                "Its suppression is bit-identical to baseline by construction. Some "
                "population was built without feature_stats' layer filter."
            )

    diagnostics = {
        "n_rows": len(ordered),
        "n_ctx_idx_checked": checked,
        "n_ctx_idx_matched": matched,
        "ctx_idx_agreement": (matched / checked) if checked else None,
        "ctx_idx_mismatch_sample": mismatches,
        "measurement_length": rhyme_ntok,
    }
    return ordered, diagnostics


def analyze_prompt(cfg: SizeConfig, slug: str, label: dict, verbose: bool = True) -> dict[str, Any]:
    """Full GPU-free analysis for one prompt. Returns the results payload that
    `run_interventions()` later extends."""
    slug_dir = cfg.graph_dir / slug
    rhyme_step = label["rhyme_step"]

    contexts = step_contexts(slug_dir)
    if not contexts:
        raise ValueError(f"no graphs in {slug_dir}")
    if rhyme_step not in contexts:
        raise ValueError(f"{slug}: rhyme_step {rhyme_step} has no graph in {slug_dir}")

    # Read once at the loosest floor the sweep will use; both views filter from it.
    step_nodes, step_tokens = load_raw_step_nodes(slug_dir, min(INFLUENCE_GRID))
    step_features, step_total = filter_at(step_nodes, INFLUENCE_THRESHOLD)

    timeline, percentiles = build_timeline(step_features, step_total)
    # Counted before the filter so the exclusion is reported, never silent.
    n_excluded_last_layer = sum(1 for k in timeline if k[0] >= cfg.n_layers - 1)
    ctx_idx_by_key = collect_ctx_idx(step_features)
    stats = feature_stats(timeline, percentiles, rhyme_step, cfg.n_layers)
    stats_by_key = {s["feat_key"]: s for s in stats}

    planning = sorted(
        (s for s in stats if s["peak_step"] < rhyme_step), key=lambda x: -x["peak_val"]
    )
    execution = sorted(
        (s for s in stats if s["peak_step"] >= rhyme_step), key=lambda x: -x["peak_val"]
    )

    rhyme_step_features = {(r["layer"], r["feat"]) for r in step_features.get(rhyme_step, [])}
    candidates = select_candidates(planning, rhyme_step_features)
    spikes = find_early_spikes(percentiles, stats_by_key)
    bands = temporal_bands(timeline, rhyme_step)

    pops = split_populations(stats, rhyme_step_features, rhyme_step)
    control = sample_matched_control(pops["candidates"], pops["rest"], RANDOM_SEED)

    # The loosest grid cell. Containment of the shipped set is *usual, not
    # guaranteed*: `rhyme_percentile` is monotonic in the influence floor, but
    # `sustain_ratio` divides by per-step totals that grow at different rates as
    # the floor drops. The measurement set is a union, so it does not depend on
    # strict containment either way.
    loose_features, loose_total = filter_at(step_nodes, min(INFLUENCE_GRID))
    loose_timeline, loose_percentiles = build_timeline(loose_features, loose_total)
    loose_stats = feature_stats(loose_timeline, loose_percentiles, rhyme_step, cfg.n_layers)
    loose_rhyme_feats = {(r["layer"], r["feat"]) for r in loose_features.get(rhyme_step, [])}
    superset = candidate_keys(
        loose_stats, loose_rhyme_feats, rhyme_step, min(PERCENTILE_GRID), min(SUSTAIN_GRID)
    )
    # `superset` was built against loose-floor normalisation, so its peak/first
    # steps can differ from the shipped view's. Positions still come from
    # `contexts`, which is threshold-independent.
    loose_ctx_idx = collect_ctx_idx(loose_features)
    ctx_idx_by_key = {
        k: {**loose_ctx_idx.get(k, {}), **ctx_idx_by_key.get(k, {})}
        for k in set(loose_ctx_idx) | set(ctx_idx_by_key)
    }

    measurement_populations = {
        "candidate": pops["candidates"],
        "superset": superset,
        "near_miss": pops["near_misses"],
        "random_control": control,
    }
    for name, entries in measurement_populations.items():
        if len(entries) > POPULATION_CEILING:
            raise RuntimeError(
                f"[{cfg.size}/{slug}] population {name!r} has {len(entries)} features, above the "
                f"sanity ceiling of {POPULATION_CEILING}. Nothing is truncated -- this is a "
                "tripwire. Check the grids and the selection predicate before raising it."
            )

    measurement_rows, position_diagnostics = build_measurement_set(
        measurement_populations, contexts, ctx_idx_by_key, rhyme_step, cfg.n_layers
    )

    if verbose:
        print(
            f"\n{'=' * 70}\n[{cfg.size}/{slug}] {label['label'].upper()}  "
            f"{label['rhyme_word']} vs {label['target_word']} @ step {rhyme_step}\n{'=' * 70}"
        )
        print(
            f"  steps={len(step_features)}  features={len(timeline)}  "
            f"planning={len(planning)}  execution={len(execution)}  candidates={len(candidates)}"
        )
        print(
            "  populations: "
            + "  ".join(f"{k}={len(v)}" for k, v in measurement_populations.items())
            + f"  -> {len(measurement_rows)} unique (layer,feat,position) rows"
        )
        print(
            f"  excluded {n_excluded_last_layer} last-layer (L{cfg.n_layers - 1}) features: "
            "no attention remains after them, so suppression is a structural no-op"
        )
        agree = position_diagnostics["ctx_idx_agreement"]
        if agree is not None:
            print(
                f"  ctx_idx agreement: {agree:.3%} of "
                f"{position_diagnostics['n_ctx_idx_checked']} checks"
            )
        for e in candidates[:TOP_N_REPORTED]:
            layer, feat = e["feat_key"]
            print(
                f"    L{layer:2d} F{feat:5d}  first=step{e['first_step']} "
                f"({rhyme_step - e['first_step']} before rhyme)  peak={e['peak_val']:.4f} "
                f"@step{e['peak_step']}  sustain={e['sustain_ratio']:.3f}"
            )
        for band_name, feats in bands.items():
            top_layers = Counter(layer for layer, _ in feats).most_common(5)
            print(f"  {band_name}: {len(feats)}  " + " ".join(f"L{ly}x{c}" for ly, c in top_layers))

    return {
        "config": {
            "size": cfg.size,
            "slug": slug,
            "model_name": cfg.model_name,
            "transcoder_repo": cfg.transcoder_repo,
            "graph_dir": str(cfg.graph_dir),
            "rhyme_step": rhyme_step,
            "rhyme_token": label["rhyme_token"],
            "rhyme_word": label["rhyme_word"],
            "target_word": label["target_word"],
            "rhyme_label": label["label"],
            "single_token_rhyme": label["single_token"],
            "influence_threshold": INFLUENCE_THRESHOLD,
            "min_rhyme_percentile": CANDIDATE_MIN_RHYME_PERCENTILE,
            "min_sustain": CANDIDATE_MIN_SUSTAIN,
            "seed": RANDOM_SEED,
            "n_layers": cfg.n_layers,
            # Pooling markers: files disagreeing on these were measured under
            # different candidate definitions. Absent means the old value.
            "excludes_last_layer": True,
            "selection_percentile_population": "measurable",
            "code_version": code_version(),
            "step_keys": list(STEP_KEYS),
            "grids": {
                "influence_threshold": list(INFLUENCE_GRID),
                "min_rhyme_percentile": list(PERCENTILE_GRID),
                "min_sustain": list(SUSTAIN_GRID),
            },
            "margins": {"sustain": SUSTAIN_MARGIN, "percentile": PERCENTILE_MARGIN},
            # Descriptive-only knobs, kept for reproducibility of the printed
            # labels. Neither is defended or swept; do not read them as findings.
            "descriptive_only": {
                "early_spike_percentile": EARLY_SPIKE_PERCENTILE,
                "early_cutoff": rhyme_step // 3,
                "mid_cutoff": (2 * rhyme_step) // 3,
            },
        },
        "statistics": {
            "n_steps": len(step_features),
            "n_unique_features": len(timeline),
            "n_excluded_last_layer": n_excluded_last_layer,
            "n_planning_features": len(planning),
            "n_execution_features": len(execution),
            "n_candidates": len(candidates),
            "n_early_spikes": len(spikes),
            "band_counts": {k: len(v) for k, v in bands.items()},
        },
        "population_counts": {
            **{k: len(v) for k, v in pops.items()},
            "superset": len(superset),
            "random_control": len(control),
            "measurement_rows": len(measurement_rows),
        },
        "position_diagnostics": position_diagnostics,
        "step_contexts": {
            str(s): {"ntok": c["ntok"], "position": c["position"], "token": c["token"]}
            for s, c in sorted(contexts.items())
        },
        "candidates": _serialize(candidates),
        "superset": _serialize(superset),
        "near_misses": _serialize(pops["near_misses"]),
        "random_control": _serialize(control),
        "early_spikes": _serialize(spikes, extra=("early_spike_step",)),
        "measurement_rows": measurement_rows,
        "steps": [
            {
                "step": s,
                "token": step_tokens[s],
                "n_features": len(step_features[s]),
                "total_influence": step_total[s],
            }
            for s in sorted(step_features)
        ],
    }


# ------------------------------------------------------------------ model half


def load_model(cfg: SizeConfig, dtype=None):
    """Load the -it model and its matching Gemma-Scope-2 transcoders.

    **Defaults to float32, not bf16, and that is a measurement decision.**
    bf16 carries 8 significand bits, so near a target logit of ~27 the
    representable spacing is 2**4 * 2**-7 = 0.125. Every logit the measurement
    reads is therefore a multiple of 0.125, and `logit_drop` inherits that grid.
    On the first real run (270M/`inspire`) this censored the result: 69 of 78
    candidates came back at *exactly* 0.0, which does not mean "no effect" but
    "smaller than the numerical resolution". A mean over a column that is 88%
    hard zeros is measuring rounding.

    fp32 costs ~2x memory for the same model. 270M and 1B fit comfortably; 4B in
    fp32 needs ~16GB of weights and wants the H100 rather than a 12GB card. Pass
    `--dtype bfloat16` to opt back out, but treat any `logit_drop` quantised to
    0.125 as a lower bound rather than a measurement.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    from circuit_tracer import ReplacementModel
    from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set

    dtype = torch.float32 if dtype is None else dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = {
        layer: hf_hub_download(
            repo_id=cfg.transcoder_repo,
            filename=f"transcoder_all/layer_{layer}_width_{WIDTH}_l0_{L0}/params.safetensors",
        )
        for layer in range(cfg.n_layers)
    }
    transcoders = load_transcoder_set(
        transcoder_paths=paths,
        scan=cfg.transcoder_repo.split("/")[-1],
        feature_input_hook="hook_resid_mid",
        feature_output_hook="hook_mlp_out",
        device=device,
        lazy_encoder=False,
        lazy_decoder=True,
        special_load_fn="gemma-scope-2",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=cfg.model_name,
        transcoders=transcoders,
        backend="transformerlens",
        dtype=dtype,
        device=device,
    )
    return model, tokenizer, device


def code_version() -> dict:
    """The commit this file was run from, plus whether the tree was dirty.

    Separate from `provenance()`, which needs a device and imports torch and so
    never runs on the `--no-interventions` path.

    Per-fix marker flags do not scale: each one distinguishes its own change and
    nothing after it. A commit hash identifies a file against any later change
    without a new flag being invented first.
    """
    import subprocess

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(REPO), *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None  # no git, no repo, or a source tarball -- not fatal
        return out.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        # A dirty tree means the commit does not fully describe the run, which
        # is worth knowing before pooling two files that name the same hash.
        "dirty": None if status is None else bool(status),
    }


def provenance(device, model_dtype=None) -> dict:
    """Hardware and library versions, written into every results file.

    bf16 carries ~8 mantissa bits and different GPU architectures select
    different kernels, so a greedy `argmax` can flip between machines and diverge
    an entire continuation. Model size is the independent variable here, which
    means hardware must not vary across sizes -- recording it makes that
    checkable after the fact instead of assumed.
    """
    import torch

    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "seed": RANDOM_SEED,
        # Recorded because it sets the resolution of every logit_drop in the
        # file: bf16 quantises to 0.125 near the target logit, fp32 does not.
        # Results measured at different dtypes must not be pooled.
        "dtype": str(getattr(model_dtype, "dtype", model_dtype)),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name()
        info["gpu_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability())
    return info


def run_interventions(model, tokenizer, device, cfg, slug, label, results) -> dict:
    """Suppress each measured feature at its own token position; read P(rhyme).

    Both suppression points (`peak_step`, `first_step`) are covered, but the
    identical `(layer, feat, position)` tuple is measured once and referenced by
    both -- see `build_measurement_set`.
    """
    from downstream_effects_addon import (
        analyze_downstream_effects,
        compute_baseline,
        generate_for,
        measure_features,
        rhyme_token_id,
    )

    results["provenance"] = provenance(device, getattr(model, "cfg", None))

    if not label["single_token"]:
        print(
            f"  [{slug}] SKIPPING interventions: rhyme token {label['rhyme_token']!r} is "
            f"multi-token, so encode(...)[0] would measure the wrong token"
        )
        results["interventions"] = {"skipped": "multi-token rhyme word"}
        return results

    rows = results["measurement_rows"]
    if not rows:
        results["interventions"] = {"skipped": "no measurable features"}
        return results

    rhyme_step = results["config"]["rhyme_step"]
    contexts = step_contexts(cfg.graph_dir / slug)
    measurement_prompt = contexts[rhyme_step]["prompt"]

    # `ensure_tokenized` is the same entry point attribution used, and it passes
    # `add_special_tokens=False`. Calling the tokenizer directly instead -- which
    # is what this pipeline used to do -- takes the default `True` and prepends a
    # second BOS to a string that already opens with a literal `<bos>`, shifting
    # every position by one. Routing through the library method is what keeps the
    # two paths from drifting apart again; it also fires the -it prefix assert.
    tokens = model.ensure_tokenized(measurement_prompt)
    expected = contexts[rhyme_step]["ntok"]
    if int(tokens.shape[0]) != expected:
        raise RuntimeError(
            f"[{cfg.size}/{slug}] re-tokenization of step {rhyme_step}'s prompt gives "
            f"{int(tokens.shape[0])} tokens but the graph recorded {expected}. Every "
            "suppression position is derived from the graph's count, so they no longer "
            "agree -- do not measure against this."
        )

    token_id = rhyme_token_id(tokenizer, label["rhyme_token"])
    baseline = compute_baseline(model, tokens, token_id)
    print(
        f"  baseline P({label['rhyme_token']!r}) = {baseline['prob']:.5f} "
        f"(logit {baseline['logit']:.3f}, rank {baseline['rank']})"
    )

    measured, failures = measure_features(model, tokens, baseline, rows, token_id)
    analyzed = analyze_downstream_effects(measured)

    # --- phase two: generations for the top slice of each step-key condition ---
    gen_rows: dict[tuple[int, int, int], dict] = {}
    per_key_top: dict[str, list[dict]] = {}
    for step_key in STEP_KEYS:
        pool = [r for r in measured if step_key in r["step_keys"]]
        pool.sort(key=lambda r: -r["logit_drop"])
        top = pool[:GENERATION_TOP_N]
        per_key_top[step_key] = [
            {"layer": r["layer"], "feat": r["feat"], "position": r["position"]} for r in top
        ]
        for r in top:
            gen_rows.setdefault((r["layer"], r["feat"], r["position"]), r)

    # One unsuppressed generation per step context actually used, so every
    # suppressed continuation has a like-for-like comparison. Generating from the
    # base prompt instead would put the position out of range: integer positions
    # do not survive into generated tokens (`_convert_open_ended_interventions`).
    steps_used = sorted({r["step"] for r in gen_rows.values()})
    baselines_by_step: dict[int, str | None] = {}
    for step in steps_used:
        baselines_by_step[step] = generate_for(model, contexts[step]["prompt"], [])
    print(f"  generating for {len(gen_rows)} features across {len(steps_used)} step contexts")

    for r in gen_rows.values():
        out = generate_for(
            model,
            contexts[r["step"]]["prompt"],
            [(r["layer"], r["position"], r["feat"], 0.0)],
        )
        if out is not None:
            r["post_intervention_output"] = out
            r["baseline_output"] = baselines_by_step.get(r["step"])

    results["interventions"] = {
        "measurement_prompt": measurement_prompt,
        "measurement_step": rhyme_step,
        "measurement_n_tokens": expected,
        "rhyme_token_id": token_id,
        "baseline": baseline,
        "baseline_generations": {str(k): v for k, v in baselines_by_step.items()},
        "n_requested": len(rows),
        "n_measured": len(measured),
        "n_failed": len(failures),
        "failures": failures,
        "generation_top_n": GENERATION_TOP_N,
        "generation_selected": per_key_top,
        "results": measured,
        "summary": {
            "all": analyzed["aggregate"],
            "by_population": {
                pop: analyze_downstream_effects([r for r in measured if pop in r["populations"]])[
                    "aggregate"
                ]
                for pop in ("candidate", "superset", "near_miss", "random_control")
            },
            "by_step_key": {
                key: analyze_downstream_effects([r for r in measured if key in r["step_keys"]])[
                    "aggregate"
                ]
                for key in STEP_KEYS
            },
        },
    }
    return results


# ------------------------------------------------------------------------ entry


def run(
    cfg: SizeConfig,
    slugs: list[str] | None = None,
    interventions: bool = True,
    dtype_name: str = "float32",
) -> None:
    if not LABELS_PATH.exists():
        raise SystemExit(f"{LABELS_PATH} missing -- run `python experiment/rhyme_labels.py` first")

    labels = {(r["size"], r["slug"]): r for r in json.loads(LABELS_PATH.read_text())}

    available = sorted(p.name for p in cfg.graph_dir.iterdir() if p.is_dir())
    targets = [s for s in (slugs or available) if s in available]
    if slugs:
        for s in slugs:
            if s not in available:
                print(f"  [{cfg.size}] {s}: no graphs -- skipping")

    model = tokenizer = device = None
    for slug in targets:
        label = labels.get((cfg.size, slug))
        if label is None:
            print(f"  [{cfg.size}] {slug}: no rhyme label -- skipping")
            continue
        if label["rhyme_step"] is None:
            print(f"  [{cfg.size}] {slug}: {label['label']} -- skipping")
            continue

        results = analyze_prompt(cfg, slug, label)

        if interventions:
            if model is None:
                import torch

                model, tokenizer, device = load_model(
                    cfg, dtype=getattr(torch, dtype_name)
                )
            results = run_interventions(model, tokenizer, device, cfg, slug, label, results)

        out = cfg.results_path(slug)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        try:
            shown = out.relative_to(EXPERIMENT)
        except ValueError:  # RESULTS_DIR overridden outside EXPERIMENT (or relative)
            shown = out
        print(f"  wrote {shown}")


def main(size: str) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=f"Circuit tracing for Gemma-3-{size}-it")
    ap.add_argument("--slugs", nargs="*", help="prompt slugs (default: all with graphs)")
    ap.add_argument(
        "--no-interventions", action="store_true", help="run the GPU-free analysis half only"
    )
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "bfloat16"),
        help=(
            "measurement precision (default float32). bf16 quantises logits to a "
            "0.125 grid at these magnitudes, which rounds most per-feature "
            "suppression effects to exactly 0 -- see load_model's docstring."
        ),
    )
    args = ap.parse_args()

    run(
        CONFIGS[size],
        slugs=args.slugs,
        interventions=not args.no_interventions,
        dtype_name=args.dtype,
    )
