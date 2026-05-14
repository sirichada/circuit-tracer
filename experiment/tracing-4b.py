import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoTokenizer
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils import create_graph_files
import torch
import torch.nn.functional as F

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

GRAPH_DIR = "./graphs/gemma-3-4b"
RHYME_TOKEN = "it"
RHYME_STEP = 9
PLANNING_WINDOW_START = 0
PLANNING_WINDOW_END = 10
INFLUENCE_THRESHOLD = 0.001
MODEL_ID = "gemma-scope-2-270m-pt"
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


# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Extract ALL features above threshold (not top-N)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Normalize influence within each step and build feature timeline
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Extract feature statistics and classify into planning/execution
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Rhyme-circuit candidates with improved filtering
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Per-step top features (normalized view)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Early spike detection: features that break high percentile early
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Temporal clustering summary
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Interventions on candidates at their peak step
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("INTERVENTION RESULTS (suppress at peak step)")
print("=" * 70)

prompt = "In the final step, everything came to it"

for candidate in candidates[:5]:
    layer, feat = candidate["feat_key"]
    peak_step = candidate["peak_step"]

    intervention = [(layer, peak_step, feat % 16384, 0.0)]
    result = model.feature_intervention_generate(prompt, intervention, max_new_tokens=20)[0]

    if "it" not in result[-30:]:
        print(f"L{layer} F{feat}: BREAKS RHYME when suppressed at step {peak_step}")
    else:
        print(f"L{layer} F{feat}: no effect")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 9 — Pseudo-CLERP: project feature decoder direction through unembedding
# ══════════════════════════════════════════════════════════════════════════════

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


print()
print("=" * 70)
print("PSEUDO-CLERP FOR TOP 15 EARLY-SPIKE FEATURES")
print("=" * 70)

for i, early_spike in enumerate(early_spikes[:15], 1):
    layer, feat = early_spike["feat_key"]
    try:
        tokens = pseudo_clerp_topk(model, layer, feat, tokenizer, top_k=10)
        print(f"\n{i:2d}. L{layer:2d} F{feat:5d}  early_spike_step={early_spike['early_spike_step']}  "
              f"sustain={early_spike['sustain_ratio']:.3f}")
        print(f"    Top tokens: {tokens}")
    except Exception as e:
        print(f"\n{i:2d}. L{layer:2d} F{feat:5d}  ERROR: {e}")


print()
print("=" * 70)
print("PSEUDO-CLERP FOR INTERVENTION-BREAKING FEATURES")
print("=" * 70)

breaking_features = [
    (16, 7915, 0, "breaks at step 0"),
    (17, 9686, 0, "breaks at step 0"),
    (17, 11499, 7, "breaks at step 7"),
]

for layer, feat, break_step, description in breaking_features:
    try:
        tokens = pseudo_clerp_topk(model, layer, feat, tokenizer, top_k=10)
        print(f"\nL{layer:2d} F{feat:5d}  {description}")
        print(f"    Top tokens: {tokens}")
    except Exception as e:
        print(f"\nL{layer:2d} F{feat:5d}  ERROR: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 10 — Dump feature statistics for downstream analysis
# ══════════════════════════════════════════════════════════════════════════════

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

with open("circuit_tracing_results.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("\nResults saved to circuit_tracing_results.json")