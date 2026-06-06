import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoTokenizer
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils import create_graph_files
import torch
import torch.nn.functional as F

from downstream_effects_addon import (
    measure_downstream_effects_batch,
    analyze_downstream_effects
)

from huggingface_hub import hf_hub_download

WIDTH = "16k"
L0 = "small"

transcoder_paths = {}
for layer in range(34):
    path = hf_hub_download(
        repo_id="google/gemma-scope-2-4b-pt",
        filename=f"transcoder_all/layer_{layer}_width_{WIDTH}_l0_{L0}/params.safetensors"
    )
    transcoder_paths[layer] = path

print(transcoder_paths)

from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set
import torch

transcoder_set = load_transcoder_set(
    transcoder_paths=transcoder_paths,
    scan="gemma-scope-2-4b-pt",
    feature_input_hook="hook_resid_mid",
    feature_output_hook="hook_mlp_out",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lazy_encoder=False,
    lazy_decoder=True,
    special_load_fn="gemma-scope-2",
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")

model = ReplacementModel.from_pretrained_and_transcoders(
    model_name="google/gemma-3-4b-pt",
    transcoders=transcoder_set,
    backend="transformerlens",
    dtype=torch.bfloat16,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

import json
import glob
import re
import requests
from collections import defaultdict, Counter
from pathlib import Path

GRAPH_DIR = "./circuit-tracer/experiment/graphs/gemma-3-4b"
RHYME_TOKEN = " crap"
RHYME_STEP = 8
PLANNING_WINDOW_START = 0
PLANNING_WINDOW_END = 7
INFLUENCE_THRESHOLD = 0.001
MODEL_ID = "gemma-scope-2-4b-pt"
FEAT_WIDTH = "16k"

# Check and find actual graph directory
import os
if not os.path.exists(GRAPH_DIR):
    print(f"Directory {GRAPH_DIR} not found. Looking for alternatives...")
    for root, dirs, files in os.walk(".", topdown=True):
        if any(f.startswith("step-") and f.endswith(".json") for f in files):
            GRAPH_DIR = root
            print(f"Found graphs in: {GRAPH_DIR}")
            break
    else:
        print("Warning: No step-*.json files found in current directory tree")
        print("Current working directory:", os.getcwd())
        print("Directory listing:")
        os.system("find . -maxdepth 3 -type d | head -20")


def parse_node_ids(node):
    """Return (layer, local_feat) from node dict, handling both id formats."""
    js = node.get("jsNodeId", "")
    m = re.match(r"^(\d+)_(\d+)-", js)
    if m:
        return int(m.group(1)), int(m.group(2))
    layer = node.get("layer")
    feat = node.get("feature")
    if layer is not None and feat is not None:
        return int(layer), int(feat)
    return None, None


# Reads all step-*.json attribution graph files from GRAPH_DIR.
# For each step/token, filters to transcoder nodes with |influence| >= INFLUENCE_THRESHOLD
# and collects (layer, feat, influence) tuples.
# Stores per-step rows in step_features_raw and total abs-influence per step in step_total_influence.

step_features_raw = {}  # step_idx -> list of all features above threshold
step_total_influence = {}  # step_idx -> sum of absolute influence values
all_step_indices = []
all_steps_tokens = {}

glob_pattern = f"{GRAPH_DIR}/step-*.json"
step_files = sorted(glob.glob(glob_pattern))
print(f"Searching for: {glob_pattern}")
print(f"Found {len(step_files)} files")
if len(step_files) == 0:
    print(f"Directory contents of {GRAPH_DIR}:")
    if os.path.exists(GRAPH_DIR):
        import subprocess
        subprocess.run(f"ls -la {GRAPH_DIR} | head -20", shell=True)
    else:
        print(f"  {GRAPH_DIR} does not exist")

for fpath in step_files:
    fname = Path(fpath).stem
    m = re.match(r"step-(\d+)-(.+)", fname)
    if not m:
        continue
    step_idx = int(m.group(1))
    token_str = m.group(2).replace("_", " ")

    with open(fpath) as f:
        data = json.load(f)

    rows = []
    step_influence_sum = 0.0
    for node in data.get("nodes", []):
        if "transcoder" not in node.get("feature_type", ""):
            continue
        inf = node.get("influence") or 0
        if inf == 0:
            continue
        abs_inf = abs(inf)
        if abs_inf < INFLUENCE_THRESHOLD:
            continue
        layer, feat = parse_node_ids(node)
        if layer is None:
            continue
        rows.append({
            "step": step_idx,
            "token": token_str,
            "layer": layer,
            "feat": feat,
            "influence": abs_inf,
            "raw_inf": inf,
        })
        step_influence_sum += abs_inf

    step_features_raw[step_idx] = rows
    step_total_influence[step_idx] = step_influence_sum
    all_step_indices.append(step_idx)
    all_steps_tokens[step_idx] = token_str

print(f"Loaded {len(step_features_raw)} steps: {sorted(step_features_raw.keys())}")
for s in sorted(step_features_raw.keys()):
    tok = all_steps_tokens[s]
    n_feats = len(step_features_raw[s])
    total_inf = step_total_influence[s]
    print(f"  step {s:02d} '{tok}': {n_feats} features (total_influence={total_inf:.4f})")


# Divides each feature's raw influence by its step's total to produce a normalized share.
# Also computes a within-step percentile rank (top feature = 100th percentile).
# Results are stored in feature_timeline[(layer, feat)][step] and feature_percentiles.

feature_timeline = defaultdict(lambda: defaultdict(float))  # (layer, feat) -> {step: norm_inf}
feature_percentiles = defaultdict(lambda: defaultdict(float))  # (layer, feat) -> {step: percentile}

for step_idx in sorted(step_features_raw.keys()):
    rows = step_features_raw[step_idx]
    step_total = step_total_influence[step_idx]

    if step_total == 0:
        continue

    sorted_rows = sorted(rows, key=lambda x: -x["influence"])

    for rank, row in enumerate(sorted_rows):
        feat_key = (row["layer"], row["feat"])
        abs_inf = row["influence"]
        norm_inf = abs_inf / step_total

        percentile = 100.0 * (len(sorted_rows) - rank) / len(sorted_rows)

        feature_timeline[feat_key][step_idx] = norm_inf
        feature_percentiles[feat_key][step_idx] = percentile

print(f"Built timeline for {len(feature_timeline)} unique features")


# For each unique feature, computes summary stats: first active step, peak step,
# peak normalized influence, influence at the rhyme step, and
# sustain_ratio = rhyme_val / peak_val (how much of peak influence persists to rhyme time).
# Features whose peak is before RHYME_STEP go into planning_features;
# those peaking at or after go into execution_features.

planning_features = []
execution_features = []
all_feature_stats = []

rhyme_step_features = set(
    (row["layer"], row["feat"]) for row in step_features_raw.get(RHYME_STEP, [])
)

for feat_key, step_inf in feature_timeline.items():
    if not step_inf:
        continue

    layer, feat = feat_key
    steps_sorted = sorted(step_inf.keys())
    first_step = steps_sorted[0]
    last_step = steps_sorted[-1]
    n_steps = len(steps_sorted)

    peak_val = max(step_inf.values())
    peak_step = max(step_inf, key=step_inf.get)
    rhyme_val = step_inf.get(RHYME_STEP, 0.0)

    peak_percentile = feature_percentiles[feat_key].get(peak_step, 0.0)
    rhyme_percentile = feature_percentiles[feat_key].get(RHYME_STEP, 0.0)

    sustain_ratio = rhyme_val / peak_val if peak_val > 0 else 0.0

    stat = {
        "feat_key": feat_key,
        "first_step": first_step,
        "peak_step": peak_step,
        "peak_val": peak_val,
        "peak_percentile": peak_percentile,
        "rhyme_val": rhyme_val,
        "rhyme_percentile": rhyme_percentile,
        "sustain_ratio": sustain_ratio,
        "n_steps": n_steps,
        "last_step": last_step,
    }
    all_feature_stats.append(stat)

    if peak_step < RHYME_STEP:
        planning_features.append(stat)
    else:
        execution_features.append(stat)

planning_features.sort(key=lambda x: -x["peak_val"])
execution_features.sort(key=lambda x: -x["peak_val"])

print("=" * 70)
print(f"PLANNING FEATURES (peak before step {RHYME_STEP}): {len(planning_features)}")
print("=" * 70)
for e in planning_features[:10]:
    l, f = e["feat_key"]
    print(f"  L{l:2d} F{f:5d}  first_active=step{e['first_step']}  peak={e['peak_val']:.4f} @ step{e['peak_step']}  "
          f"rhyme={e['rhyme_val']:.4f}  sustain_ratio={e['sustain_ratio']:.4f}  percentile_at_rhyme={e['rhyme_percentile']:.1f}%")

print()
print("=" * 70)
print(f"EXECUTION FEATURES (peak at step {RHYME_STEP} or after): {len(execution_features)}")
print("=" * 70)
for e in execution_features[:10]:
    l, f = e["feat_key"]
    print(f"  L{l:2d} F{f:5d}  first_active=step{e['first_step']}  peak={e['peak_val']:.4f} @ step{e['peak_step']}  "
          f"rhyme={e['rhyme_val']:.4f}  sustain_ratio={e['sustain_ratio']:.4f}  percentile_at_rhyme={e['rhyme_percentile']:.1f}%")


# Filters planning_features down to the strongest rhyme-circuit candidates.
# A feature qualifies only if it is also active at the rhyme step (>= 50th percentile there)
# and has sustain_ratio >= 0.3 — meaning it activates early and stays influential at rhyme time.

print()
print("=" * 70)
print("RHYME-CIRCUIT CANDIDATES (improved filtering)")
print("=" * 70)

candidates = []
for e in planning_features:
    if e["feat_key"] not in rhyme_step_features:
        continue
    if e["rhyme_percentile"] < 50:
        continue
    if e["sustain_ratio"] < 0.3:
        continue

    candidates.append(e)

candidates.sort(key=lambda x: -(x["peak_val"] + x["rhyme_val"]))

for e in candidates[:15]:
    l, f = e["feat_key"]
    steps_before = RHYME_STEP - e["first_step"]
    print(f"  L{l:2d} F{f:5d}  first_active=step{e['first_step']} ({steps_before} steps before rhyme)  "
          f"peak={e['peak_val']:.4f} @ step{e['peak_step']}  rhyme={e['rhyme_val']:.4f}  "
          f"sustain={e['sustain_ratio']:.3f}  rhyme_percentile={e['rhyme_percentile']:.1f}%")

print(f"\nTotal rhyme-circuit candidates: {len(candidates)}")


# Sanity-check / qualitative view: prints the top 5 features by normalized influence
# for every step, along with each feature's within-step percentile rank.
# The rhyme step is annotated with an arrow for easy identification.

print()
print("=" * 70)
print("PER-STEP TOP FEATURES (normalized influence + percentile rank)")
print("=" * 70)

for step_idx in sorted(all_step_indices):
    rows = step_features_raw[step_idx]
    tok = all_steps_tokens[step_idx]
    marker = "  ← RHYME" if step_idx == RHYME_STEP else ""

    rows_sorted = sorted(rows, key=lambda x: -x["influence"])
    step_total = step_total_influence[step_idx]

    print(f"\nstep {step_idx:02d} '{tok}'{marker}")
    for r in rows_sorted[:5]:
        feat_key = (r["layer"], r["feat"])
        norm_inf = r["influence"] / step_total if step_total > 0 else 0
        percentile = feature_percentiles[feat_key].get(step_idx, 0)
        print(f"    L{r['layer']:2d} F{r['feat']:5d}  norm_inf={norm_inf:.4f}  percentile={percentile:.1f}%")


# Finds features that first reach >= 70th percentile prominence very early in the sequence.
# Results are sorted by early_spike_step so the features that become prominent soonest
# appear first — regardless of whether they are rhyme-circuit candidates.

print()
print("=" * 70)
print("EARLY SPIKE DETECTION (first step at ≥70th percentile)")
print("=" * 70)

early_spikes = []
for feat_key, step_percentiles in feature_percentiles.items():
    steps_sorted = sorted(step_percentiles.keys())
    for step in steps_sorted:
        if step_percentiles[step] >= 70.0:
            early_spike_step = step
            break
    else:
        continue

    feature_stat = next((s for s in all_feature_stats if s["feat_key"] == feat_key), None)
    if feature_stat:
        early_spikes.append({
            "feat_key": feat_key,
            "early_spike_step": early_spike_step,
            "peak_step": feature_stat["peak_step"],
            "peak_val": feature_stat["peak_val"],
            "rhyme_val": feature_stat["rhyme_val"],
            "rhyme_percentile": feature_stat["rhyme_percentile"],
            "sustain_ratio": feature_stat["sustain_ratio"],
        })

early_spikes.sort(key=lambda x: x["early_spike_step"])

for e in early_spikes[:15]:
    l, f = e["feat_key"]
    print(f"  L{l:2d} F{f:5d}  early_spike @ step{e['early_spike_step']}  "
          f"peak={e['peak_val']:.4f} @ step{e['peak_step']}  "
          f"rhyme={e['rhyme_val']:.4f}  sustain={e['sustain_ratio']:.3f}")

print(f"\nTotal features with early spike: {len(early_spikes)}")


# Splits all features into three temporal bands based on where each feature's peak falls:
#   EARLY  (steps 0 – RHYME_STEP//3)
#   MID    (steps RHYME_STEP//3 – 2*RHYME_STEP//3)
#   LATE   (steps 2*RHYME_STEP//3 – end)
# For each band, reports feature count, most active layers, and the top 3 features by peak influence.

print()
print("=" * 70)
print("TEMPORAL CLUSTERING SUMMARY")
print("=" * 70)

early_cutoff = RHYME_STEP // 3
mid_cutoff = (2 * RHYME_STEP) // 3

bands = {"EARLY (planning)": [], "MID (buildup)": [], "LATE (execution)": []}

for feat_key, step_inf in feature_timeline.items():
    if not step_inf:
        continue
    peak_step = max(step_inf, key=step_inf.get)
    if peak_step <= early_cutoff:
        bands["EARLY (planning)"].append(feat_key)
    elif peak_step <= mid_cutoff:
        bands["MID (buildup)"].append(feat_key)
    else:
        bands["LATE (execution)"].append(feat_key)

for band_name, feats in bands.items():
    layer_counts = Counter(layer for layer, _ in feats)
    top_layers = layer_counts.most_common(5)
    print(f"\n{band_name}: {len(feats)} features")
    print(f"  Top layers: " + "  ".join(f"L{l}×{c}" for l, c in top_layers))

    top_in_band = sorted(
        feats,
        key=lambda lf: max(feature_timeline[lf].values()),
        reverse=True
    )[:3]
    for lf in top_in_band:
        peak_inf = max(feature_timeline[lf].values())
        peak_s = max(feature_timeline[lf], key=feature_timeline[lf].get)
        print(f"    L{lf[0]:2d} F{lf[1]:5d}  peak_norm_inf={peak_inf:.4f} @ step{peak_s}")

# Packages config, aggregate statistics, the candidates list, and the early_spikes list
# into output_data. This dict is extended with downstream intervention results
# and written to circuit_tracing_results_4b.json in the next section.

output_data = {
    "config": {
        "graph_dir": GRAPH_DIR,
        "rhyme_step": RHYME_STEP,
        "influence_threshold": INFLUENCE_THRESHOLD,
        "early_cutoff": early_cutoff,
        "mid_cutoff": mid_cutoff,
    },
    "statistics": {
        "n_steps": len(all_step_indices),
        "n_unique_features": len(feature_timeline),
        "n_planning_features": len(planning_features),
        "n_execution_features": len(execution_features),
        "n_candidates": len(candidates),
        "n_early_spikes": len(early_spikes),
    },
    "candidates": [
        {
            "layer": c["feat_key"][0],
            "feat": c["feat_key"][1],
            "first_step": c["first_step"],
            "peak_step": c["peak_step"],
            "peak_val": float(c["peak_val"]),
            "rhyme_val": float(c["rhyme_val"]),
            "sustain_ratio": float(c["sustain_ratio"]),
            "rhyme_percentile": float(c["rhyme_percentile"]),
        }
        for c in candidates
    ],
    "early_spikes": [
        {
            "layer": e["feat_key"][0],
            "feat": e["feat_key"][1],
            "early_spike_step": e["early_spike_step"],
            "peak_step": e["peak_step"],
            "peak_val": float(e["peak_val"]),
            "rhyme_val": float(e["rhyme_val"]),
            "sustain_ratio": float(e["sustain_ratio"]),
        }
        for e in early_spikes
    ],
}

# Causal intervention experiment: for each rhyme-circuit candidate, zeroes out the feature
# at two different points — its peak step and its first active step — then measures how much
# the model's probability of generating the rhyme token drops.
# Prints full before/after generations for the top 10 features by probability drop,
# compares the two suppression strategies side by side, and serializes all results
# to circuit_tracing_results_4b.json.

print()
print("=" * 70)
print("DOWNSTREAM EFFECTS MEASUREMENT (suppress at peak step)")
print("=" * 70)
 
prompt = "A rhyming couplet:\nHe saw a carrot and had to grab it,\n"
measurement_prompt = "A rhyming couplet:\nHe saw a carrot and had to grab it,\nHe ate it and then he had to"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
print("\nGenerating baseline output without intervention...")
baseline_output = model.feature_intervention_generate(prompt, [], do_sample=False)[0]
print(f"Baseline: {baseline_output}")

# Check what model predicts just before "crap"
prompt_before_crap = "A rhyming couplet:\nHe saw a carrot and had to grab it,\nHe ate it and then he had to"
with torch.no_grad():
    logits, _ = model.feature_intervention(prompt_before_crap, [])
    last_logits = logits[0, -1, :].float()
    probs = torch.softmax(last_logits, dim=-1)
    top5_probs, top5_ids = probs.topk(5)
    print("Predicting after 'had to':")
    for i in range(5):
        token = tokenizer.convert_ids_to_tokens(top5_ids[i].item())
        print(f"  {token!r}: {top5_probs[i].item():.4f}")
    
    crap_id = 54122
    rank = (probs > probs[crap_id]).sum().item()
    print(f"\n  ' crap' rank: {rank}, prob: {probs[crap_id].item():.6f}")

with torch.no_grad():
    logits, _ = model.feature_intervention(prompt, [])
    print(f"logits shape: {logits.shape}")  # confirm full shape
    last_logits = logits[-1, -1, :]  # last batch, last seq position
    probs = torch.softmax(last_logits.float(), dim=-1)
    top5_probs, top5_ids = probs.topk(5)
    
    for i in range(5):
        prob = top5_probs[i].item()
        idx = top5_ids[i].item()
        token = tokenizer.convert_ids_to_tokens(idx)
        print(f"  {token!r}: {prob:.4f}")
    
    crap_id = 54122
    rank = (probs > probs[crap_id]).sum().item()
    print(f'\n  " crap" rank: {rank}, prob: {probs[crap_id].item():.6f}')

downstream_results = measure_downstream_effects_batch(
    model=model,
    prompt=prompt,
    candidates_list=candidates,
    suppression_step_key='peak_step',
    device=device,
    tokenizer=tokenizer,
    RHYME_TOKEN=RHYME_TOKEN,
    max_new_tokens=20,
    max_candidates=50,
    measurement_prompt=measurement_prompt
)
 
analyzed_results = analyze_downstream_effects(downstream_results, top_n=20)
 
print("\n" + "=" * 100)
print("POST-INTERVENTION OUTPUTS FOR TOP FEATURES BY PROBABILITY DROP")
print("=" * 100)
 
sorted_by_prob_drop = sorted(downstream_results, key=lambda x: -x['prob_drop'])
 
for i, result in enumerate(sorted_by_prob_drop[:10], 1):
    layer = result['layer']
    feat = result['feat']
    suppression_step = result['suppression_step']
    
    intervention = [(layer, suppression_step, feat, 0.0)]
    post_intervention_output = model.feature_intervention_generate(prompt, intervention, max_new_tokens=20)[0]
    
    print(f"\n{i}. L{layer:2d} F{feat:5d} (suppress @ step {suppression_step:2d})")
    print(f"   P(rhyme): {result['original_prob']:.4f} → {result['suppressed_prob']:.4f}  (drop: {result['prob_drop_pct']:.1f}%)")
    print(f"   Rank: {result['original_rank']:3d} → {result['suppressed_rank']:3d}  (shift: {result['rank_shift']:+4d})")
    print(f"   Baseline:         {baseline_output}")
    print(f"   Post-intervention: {post_intervention_output}")
 
 
print("\n" + "=" * 100)
print("MEASURING DOWNSTREAM EFFECTS - SUPPRESSION AT FIRST STEP")
print("=" * 100)
 
downstream_results_first = measure_downstream_effects_batch(
    model=model,
    prompt=prompt,
    candidates_list=candidates,
    suppression_step_key='first_step',
    device=device,
    tokenizer=tokenizer,
    RHYME_TOKEN=RHYME_TOKEN,
    max_new_tokens=20,
    max_candidates=30,
    measurement_prompt=measurement_prompt
)
 
analyzed_results_first = analyze_downstream_effects(downstream_results_first, top_n=15)
 
print("\n" + "=" * 100)
print("POST-INTERVENTION OUTPUTS FOR TOP FEATURES (FIRST STEP SUPPRESSION)")
print("=" * 100)

sorted_by_prob_drop_first = sorted(downstream_results_first, key=lambda x: -x['prob_drop'])
 
for i, result in enumerate(sorted_by_prob_drop_first[:8], 1):
    layer = result['layer']
    feat = result['feat']
    suppression_step = result['suppression_step']
    
    intervention = [(layer, suppression_step, feat, 0.0)]
    post_intervention_output = model.feature_intervention_generate(prompt, intervention, max_new_tokens=20)[0]
    
    print(f"\n{i}. L{layer:2d} F{feat:5d} (suppress @ step {suppression_step:2d})")
    print(f"   P(rhyme): {result['original_prob']:.4f} → {result['suppressed_prob']:.4f}  (drop: {result['prob_drop_pct']:.1f}%)")
    print(f"   Rank: {result['original_rank']:3d} → {result['suppressed_rank']:3d}  (shift: {result['rank_shift']:+4d})")
    print(f"   Baseline:         {baseline_output}")
    print(f"   Post-intervention: {post_intervention_output}")
 
 
print("\n" + "=" * 100)
print("COMPARING SUPPRESSION STRATEGIES")
print("=" * 100)
 
feature_set_peak = set((r['layer'], r['feat']) for r in downstream_results)
feature_set_first = set((r['layer'], r['feat']) for r in downstream_results_first)
shared_features = feature_set_peak & feature_set_first
 
print(f"Features tested with peak_step suppression: {len(feature_set_peak)}")
print(f"Features tested with first_step suppression: {len(feature_set_first)}")
print(f"Features tested with both strategies: {len(shared_features)}")
 
if shared_features:
    print("\nComparison of suppression strategies (shared features):")
    
    peak_results_map = {(r['layer'], r['feat']): r for r in downstream_results}
    first_results_map = {(r['layer'], r['feat']): r for r in downstream_results_first}
    
    comparisons = []
    for layer, feat in sorted(list(shared_features)):
        peak_r = peak_results_map[(layer, feat)]
        first_r = first_results_map[(layer, feat)]
        
        comparisons.append({
            'layer': layer,
            'feat': feat,
            'peak_prob_drop': peak_r['prob_drop'],
            'first_prob_drop': first_r['prob_drop'],
            'peak_rank_shift': peak_r['rank_shift'],
            'first_rank_shift': first_r['rank_shift'],
        })
    
    comparisons_sorted = sorted(comparisons, key=lambda x: -(abs(x['peak_prob_drop'] - x['first_prob_drop'])))
    
    for i, comp in enumerate(comparisons_sorted[:15], 1):
        print(f"{i:2d}. L{comp['layer']:2d} F{comp['feat']:5d}")
        print(f"    Peak step:  prob_drop={comp['peak_prob_drop']:.4f}  rank_shift={comp['peak_rank_shift']:+4d}")
        print(f"    First step: prob_drop={comp['first_prob_drop']:.4f}  rank_shift={comp['first_rank_shift']:+4d}")
        print(f"    Difference: prob_drop_delta={abs(comp['peak_prob_drop'] - comp['first_prob_drop']):.4f}")
 
 
# Force ensure everything is a standard Python primitive before dumping
def sanitize_for_json(obj):
    if hasattr(obj, 'item'): # Catches PyTorch/NumPy scalars
        return obj.item()
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    return obj

try:
    output_data_with_downstream = output_data.copy()
    output_data_with_downstream["downstream_effects"] = {
        "peak_step_suppression": {
            "results_count": len(downstream_results),
            "aggregate": sanitize_for_json(analyzed_results.get("aggregate", {})),
            "top_by_prob_drop": [
                {
                    "layer": int(r["layer"]),
                    "feat": int(r["feat"]),
                    "prob_drop": float(r["prob_drop"]),
                    "prob_drop_pct": float(r["prob_drop_pct"]),
                    "rank_shift": int(r["rank_shift"]),
                }
                for r in analyzed_results.get('sorted_by_prob_drop', [])[:30]
            ]
        },
        "first_step_suppression": {
            "results_count": len(downstream_results_first),
            "aggregate": sanitize_for_json(analyzed_results_first.get("aggregate", {})),
            "top_by_prob_drop": [
                {
                    "layer": int(r["layer"]),
                    "feat": int(r["feat"]),
                    "prob_drop": float(r["prob_drop"]),
                    "prob_drop_pct": float(r["prob_drop_pct"]),
                    "rank_shift": int(r["rank_shift"]),
                }
                for r in analyzed_results_first.get('sorted_by_prob_drop', [])[:30]
            ]
        }
    }
    
    with open("circuit-tracer/experiment/tracing/circuit_tracing_results_4b.json", "w") as f:
        json.dump(output_data_with_downstream, f, indent=2)
     
    print("\nCombined results successfully saved to tracing/circuit_tracing_results_4b.json")

except Exception as e:
    print(f"\nERROR: Failed to write JSON file: {e}")
    # Fallback: save what we have so we don't lose the whole script's progress
    with open("circuit-tracer/experiment/tracing/circuit_tracing_results_4b_fallback.json", "w") as f:
        json.dump({"status": "failed_downstream_formatting", "config": output_data.get("config")}, f)


# Pseudo-CLERP: for each top candidate and early-spike feature, multiplies the transcoder's
# decoder weight vector W_dec[feat] by the unembedding matrix W_U to get logits over
# the vocabulary — a quick proxy for "what tokens does this feature predict?"
# Also runs the probe on any features that caused anomalous behavior during intervention
# (errors, extreme rank shifts, or near-zero suppressed probability).

def pseudo_clerp_topk(model, layer, local_feat, tokenizer, top_k=10):
    """
    Project feature decoder direction through unembedding matrix.
    Returns top-k predicted tokens for this feature.
    """
    tc = model.transcoders[layer]
    W_dec = tc.W_dec[local_feat]
    
    with torch.no_grad():
        W_U = model.unembed.W_U
        logits = torch.matmul(W_U.T, W_dec.to(W_U.dtype))
    
    top_ids = logits.topk(top_k).indices.tolist()
    top_tokens = [tokenizer.decode([i]) for i in top_ids]
    
    return top_tokens


print()
print("=" * 70)
print("PSEUDO-CLERP FOR TOP 10 RHYME-CIRCUIT CANDIDATES")
print("=" * 70)

for i, candidate in enumerate(candidates[:10], 1):
    layer, feat = candidate["feat_key"]
    try:
        tokens = pseudo_clerp_topk(model, layer, feat, tokenizer, top_k=10)
        print(f"\n{i:2d}. L{layer:2d} F{feat:5d}  first_active=step{candidate['first_step']}  "
              f"sustain={candidate['sustain_ratio']:.3f}")
        print(f"    Top tokens: {tokens}")
    except Exception as e:
        print(f"\n{i:2d}. L{layer:2d} F{feat:5d}  ERROR: {e}")


import sys

print("=" * 70, flush=True)
print("PSEUDO-CLERP FOR TOP 15 EARLY-SPIKE FEATURES", flush=True)
print("=" * 70, flush=True)

for i, early_spike in enumerate(early_spikes[:15], 1):
    layer, feat = early_spike["feat_key"]
    try:
        tokens = pseudo_clerp_topk(model, layer, feat, tokenizer, top_k=10)
        print(f"\n{i:2d}. L{layer:2d} F{feat:5d}  early_spike_step={early_spike['early_spike_step']}  "
              f"sustain={early_spike['sustain_ratio']:.3f}")
        print(f"    Top tokens: {tokens}", flush=True)
    except Exception as e:
        print(f"\n{i:2d}. L{layer:2d} F{feat:5d}  ERROR: {e}", flush=True)

# Explicit cleanup to prevent __del__ racing with stdout at shutdown
del model
torch.cuda.empty_cache()
sys.stdout.flush()