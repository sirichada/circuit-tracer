import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoTokenizer
from circuit_tracer import ReplacementModel
import torch
import json
import glob
import re
from collections import defaultdict
from pathlib import Path
from huggingface_hub import hf_hub_download
from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set

# ── Config ────────────────────────────────────────────────────────────────────

GRAPH_DIR = "./circuit-tracer/experiment/graphs/gemma-3-4b"
RESULTS_PATH = "circuit-tracer/experiment/tracing/circuit_tracing_results_4b.json"
RHYME_TOKEN = " crap"
RHYME_STEP = 8
INFLUENCE_THRESHOLD = 0.001
PROMPT = "A rhyming couplet:\nHe saw a carrot and had to grab it,\n"
MEASUREMENT_PROMPT = "A rhyming couplet:\nHe saw a carrot and had to grab it,\nHe ate it and then he had to"

# Candidate filter thresholds (must match tracing-4b.py)
SUSTAIN_THRESHOLD = 0.3
PERCENTILE_THRESHOLD = 50

# Near-miss margins: how far below the threshold to look
SUSTAIN_MARGIN = 0.15   # catches sustain_ratio in [0.15, 0.30)
PERCENTILE_MARGIN = 15  # catches rhyme_percentile in [35, 50)

# ── Locate graph directory ────────────────────────────────────────────────────

if not os.path.exists(GRAPH_DIR):
    for root, dirs, files in os.walk(".", topdown=True):
        if any(f.startswith("step-") and f.endswith(".json") for f in files):
            GRAPH_DIR = root
            print(f"Found graphs in: {GRAPH_DIR}")
            break
    else:
        raise FileNotFoundError("No step-*.json files found")

# ── Load model ────────────────────────────────────────────────────────────────

WIDTH = "16k"
L0 = "small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transcoder_paths = {}
for layer in range(34):
    transcoder_paths[layer] = hf_hub_download(
        repo_id="google/gemma-scope-2-4b-pt",
        filename=f"transcoder_all/layer_{layer}_width_{WIDTH}_l0_{L0}/params.safetensors"
    )

transcoder_set = load_transcoder_set(
    transcoder_paths=transcoder_paths,
    scan="gemma-scope-2-4b-pt",
    feature_input_hook="hook_resid_mid",
    feature_output_hook="hook_mlp_out",
    device=device,
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
    device=device,
)

# ── Re-run feature extraction (same as tracing-4b.py) ────────────────────────

def parse_node_ids(node):
    js = node.get("jsNodeId", "")
    m = re.match(r"^(\d+)_(\d+)-", js)
    if m:
        return int(m.group(1)), int(m.group(2))
    layer = node.get("layer")
    feat = node.get("feature")
    if layer is not None and feat is not None:
        return int(layer), int(feat)
    return None, None

step_features_raw = {}
step_total_influence = {}

for fpath in sorted(glob.glob(f"{GRAPH_DIR}/step-*.json")):
    fname = Path(fpath).stem
    m = re.match(r"step-(\d+)-(.+)", fname)
    if not m:
        continue
    step_idx = int(m.group(1))

    with open(fpath) as f:
        data = json.load(f)

    rows = []
    step_influence_sum = 0.0
    for node in data.get("nodes", []):
        if "transcoder" not in node.get("feature_type", ""):
            continue
        inf = node.get("influence") or 0
        if inf == 0 or abs(inf) < INFLUENCE_THRESHOLD:
            continue
        layer, feat = parse_node_ids(node)
        if layer is None:
            continue
        rows.append({"layer": layer, "feat": feat, "influence": abs(inf)})
        step_influence_sum += abs(inf)

    step_features_raw[step_idx] = rows
    step_total_influence[step_idx] = step_influence_sum

feature_timeline = defaultdict(lambda: defaultdict(float))
feature_percentiles = defaultdict(lambda: defaultdict(float))

for step_idx, rows in step_features_raw.items():
    step_total = step_total_influence[step_idx]
    if step_total == 0:
        continue
    sorted_rows = sorted(rows, key=lambda x: -x["influence"])
    for rank, row in enumerate(sorted_rows):
        feat_key = (row["layer"], row["feat"])
        feature_timeline[feat_key][step_idx] = row["influence"] / step_total
        feature_percentiles[feat_key][step_idx] = 100.0 * (len(sorted_rows) - rank) / len(sorted_rows)

rhyme_step_features = set(
    (r["layer"], r["feat"]) for r in step_features_raw.get(RHYME_STEP, [])
)

# ── Identify near-miss features ───────────────────────────────────────────────
# A near-miss is a planning feature (peak before RHYME_STEP) that is present at
# the rhyme step but failed one or both filters by a small margin.

print("=" * 70)
print("NEAR-MISS FEATURES")
print(f"  sustain_ratio in [{SUSTAIN_THRESHOLD - SUSTAIN_MARGIN:.2f}, {SUSTAIN_THRESHOLD})")
print(f"  OR rhyme_percentile in [{PERCENTILE_THRESHOLD - PERCENTILE_MARGIN}, {PERCENTILE_THRESHOLD})")
print("=" * 70)

near_misses = []

for feat_key, step_inf in feature_timeline.items():
    if not step_inf:
        continue

    peak_step = max(step_inf, key=step_inf.get)
    if peak_step >= RHYME_STEP:
        continue  # not a planning feature

    if feat_key not in rhyme_step_features:
        continue  # not active at rhyme step at all

    peak_val = step_inf[peak_step]
    rhyme_val = step_inf.get(RHYME_STEP, 0.0)
    sustain_ratio = rhyme_val / peak_val if peak_val > 0 else 0.0
    rhyme_percentile = feature_percentiles[feat_key].get(RHYME_STEP, 0.0)

    passed_sustain = sustain_ratio >= SUSTAIN_THRESHOLD
    passed_percentile = rhyme_percentile >= PERCENTILE_THRESHOLD

    if passed_sustain and passed_percentile:
        continue  # this is a full candidate, not a near-miss

    # Near-miss: failed at least one filter, but within margin on both
    sustain_in_margin = (SUSTAIN_THRESHOLD - SUSTAIN_MARGIN) <= sustain_ratio < SUSTAIN_THRESHOLD
    percentile_in_margin = (PERCENTILE_THRESHOLD - PERCENTILE_MARGIN) <= rhyme_percentile < PERCENTILE_THRESHOLD

    # Must be within margin on any failed filter (and at least passing on the other)
    failed_only_sustain = not passed_sustain and passed_percentile and sustain_in_margin
    failed_only_percentile = passed_sustain and not passed_percentile and percentile_in_margin
    failed_both_in_margin = not passed_sustain and not passed_percentile and sustain_in_margin and percentile_in_margin

    if not (failed_only_sustain or failed_only_percentile or failed_both_in_margin):
        continue

    near_misses.append({
        "feat_key": feat_key,
        "layer": feat_key[0],
        "feat": feat_key[1],
        "peak_step": peak_step,
        "peak_val": peak_val,
        "rhyme_val": rhyme_val,
        "sustain_ratio": sustain_ratio,
        "rhyme_percentile": rhyme_percentile,
        "failed_sustain": not passed_sustain,
        "failed_percentile": not passed_percentile,
        "first_step": min(step_inf.keys()),
    })

near_misses.sort(key=lambda x: -(x["sustain_ratio"] + x["rhyme_percentile"] / 100))

print(f"Found {len(near_misses)} near-miss features\n")
for e in near_misses[:20]:
    failed = []
    if e["failed_sustain"]:
        failed.append(f"sustain={e['sustain_ratio']:.3f} < {SUSTAIN_THRESHOLD}")
    if e["failed_percentile"]:
        failed.append(f"percentile={e['rhyme_percentile']:.1f} < {PERCENTILE_THRESHOLD}")
    print(f"  L{e['layer']:2d} F{e['feat']:5d}  peak={e['peak_val']:.4f}@step{e['peak_step']}  "
          f"rhyme={e['rhyme_val']:.4f}  failed: {' | '.join(failed)}")

# ── Run interventions on near-miss features ───────────────────────────────────

print()
print("=" * 70)
print("INTERVENTIONS ON NEAR-MISS FEATURES (suppress at peak step)")
print("=" * 70)

with torch.no_grad():
    logits, _ = model.feature_intervention(MEASUREMENT_PROMPT, [])
    last_logits = logits[0, -1, :].float()
    probs_baseline = torch.softmax(last_logits, dim=-1)
    rhyme_token_ids = tokenizer.encode(RHYME_TOKEN, add_special_tokens=False)
    rhyme_id = rhyme_token_ids[-1]
    original_prob = probs_baseline[rhyme_id].item()
    original_rank = (probs_baseline > probs_baseline[rhyme_id]).sum().item()

print(f"Baseline — P('{RHYME_TOKEN}'): {original_prob:.4f}  rank: {original_rank}")

baseline_output = model.feature_intervention_generate(PROMPT, [], do_sample=False)[0]
print(f"Baseline generation: {baseline_output}\n")

near_miss_results = []

for e in near_misses[:30]:
    layer = e["layer"]
    feat = e["feat"]
    suppression_step = e["peak_step"]

    try:
        intervention = [(layer, suppression_step, feat, 0.0)]

        with torch.no_grad():
            logits, _ = model.feature_intervention(MEASUREMENT_PROMPT, intervention)
            last_logits = logits[0, -1, :].float()
            probs = torch.softmax(last_logits, dim=-1)
            suppressed_prob = probs[rhyme_id].item()
            suppressed_rank = (probs > probs[rhyme_id]).sum().item()

        prob_drop = original_prob - suppressed_prob
        prob_drop_pct = 100 * prob_drop / original_prob if original_prob > 0 else 0
        rank_shift = suppressed_rank - original_rank

        near_miss_results.append({
            **e,
            "suppression_step": suppression_step,
            "original_prob": original_prob,
            "suppressed_prob": suppressed_prob,
            "prob_drop": prob_drop,
            "prob_drop_pct": prob_drop_pct,
            "rank_shift": rank_shift,
        })

    except Exception as ex:
        print(f"  L{layer:2d} F{feat:5d}  ERROR: {ex}")

# ── Results ───────────────────────────────────────────────────────────────────

near_miss_results.sort(key=lambda x: -x["prob_drop"])

print("Near-miss results sorted by probability drop:\n")
for r in near_miss_results:
    failed = []
    if r["failed_sustain"]:
        failed.append(f"sustain={r['sustain_ratio']:.3f}")
    if r["failed_percentile"]:
        failed.append(f"pct={r['rhyme_percentile']:.1f}")
    print(f"  L{r['layer']:2d} F{r['feat']:5d}  "
          f"P(rhyme): {r['original_prob']:.4f} → {r['suppressed_prob']:.4f}  "
          f"drop: {r['prob_drop_pct']:.1f}%  rank_shift: {r['rank_shift']:+d}  "
          f"failed: {' | '.join(failed)}")

# ── Compare against actual candidates ────────────────────────────────────────

print()
print("=" * 70)
print("COMPARISON: NEAR-MISSES vs CANDIDATES")
print("=" * 70)

with open(RESULTS_PATH) as f:
    saved = json.load(f)

candidate_drops = [r["prob_drop"] for r in saved["downstream_effects"]["peak_step_suppression"]["top_by_prob_drop"]]
near_miss_drops = [r["prob_drop"] for r in near_miss_results]

def stats(drops, label):
    if not drops:
        print(f"{label}: no data")
        return
    nonzero = [d for d in drops if d > 0]
    print(f"{label} (n={len(drops)}):")
    print(f"  max drop:      {max(drops):.4f}")
    print(f"  mean drop:     {sum(drops)/len(drops):.4f}")
    print(f"  positive drops: {len(nonzero)} / {len(drops)}")

stats(candidate_drops, "Candidates (from saved results)")
print()
stats(near_miss_drops, "Near-misses (newly tested)")

# Features where near-miss had a meaningful drop (> 10% of baseline probability)
meaningful_threshold = 0.1 * original_prob
meaningful_near_misses = [r for r in near_miss_results if r["prob_drop"] > meaningful_threshold]

print(f"\nNear-misses with prob_drop > {meaningful_threshold:.4f} (10% of baseline): {len(meaningful_near_misses)}")
if meaningful_near_misses:
    print("These features may warrant lowering the filter thresholds:\n")
    for r in meaningful_near_misses:
        post_output = model.feature_intervention_generate(
            PROMPT, [(r["layer"], r["suppression_step"], r["feat"], 0.0)], max_new_tokens=20
        )[0]
        failed = []
        if r["failed_sustain"]:
            failed.append(f"sustain={r['sustain_ratio']:.3f} < {SUSTAIN_THRESHOLD}")
        if r["failed_percentile"]:
            failed.append(f"percentile={r['rhyme_percentile']:.1f} < {PERCENTILE_THRESHOLD}")
        print(f"  L{r['layer']:2d} F{r['feat']:5d}  drop={r['prob_drop_pct']:.1f}%  failed: {' | '.join(failed)}")
        print(f"    Baseline:          {baseline_output}")
        print(f"    Post-intervention: {post_output}")