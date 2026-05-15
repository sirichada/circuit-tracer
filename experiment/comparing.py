import json
from collections import defaultdict, Counter
from pathlib import Path

def load_results(filepath):
    with open(filepath) as f:
        return json.load(f)

def extract_features_from_candidates(data):
    return set((c['layer'], c['feat']) for c in data['candidates'])

def extract_features_from_early_spikes(data):
    return set((e['layer'], e['feat']) for e in data['early_spikes'])

def build_candidate_map(data):
    return {(c['layer'], c['feat']): c for c in data['candidates']}

def build_early_spike_map(data):
    return {(e['layer'], e['feat']): e for e in data['early_spikes']}

results_270m = load_results('/teamspace/studios/this_studio/circuit-tracer/experiment/tracing/circuit_tracing_results_270m.json')
results_1b = load_results('/teamspace/studios/this_studio/circuit-tracer/experiment/tracing/circuit_tracing_results_1b.json')
results_4b = load_results('/teamspace/studios/this_studio/circuit-tracer/experiment/tracing/circuit_tracing_results_4b.json')

models = {
    '270m': results_270m,
    '1b': results_1b,
    '4b': results_4b,
}

model_order = ['270m', '1b', '4b']

features_candidates = {name: extract_features_from_candidates(data) for name, data in models.items()}
features_early_spikes = {name: extract_features_from_early_spikes(data) for name, data in models.items()}
candidate_maps = {name: build_candidate_map(data) for name, data in models.items()}
early_spike_maps = {name: build_early_spike_map(data) for name, data in models.items()}

print("=" * 80)
print("INSIGHT 1: FEATURE PERSISTENCE ACROSS MODEL SCALES")
print("=" * 80)

features_in_all = features_candidates['270m'] & features_candidates['1b'] & features_candidates['4b']
features_in_270m_only = features_candidates['270m'] - features_candidates['1b'] - features_candidates['4b']
features_in_1b_only = features_candidates['1b'] - features_candidates['270m'] - features_candidates['4b']
features_in_4b_only = features_candidates['4b'] - features_candidates['270m'] - features_candidates['1b']
features_in_270m_1b = (features_candidates['270m'] & features_candidates['1b']) - features_candidates['4b']
features_in_270m_4b = (features_candidates['270m'] & features_candidates['4b']) - features_candidates['1b']
features_in_1b_4b = (features_candidates['1b'] & features_candidates['4b']) - features_candidates['270m']

print(f"Features in all three models: {len(features_in_all)}")
print(f"Features in 270m only: {len(features_in_270m_only)}")
print(f"Features in 1b only: {len(features_in_1b_only)}")
print(f"Features in 4b only: {len(features_in_4b_only)}")
print(f"Features in 270m & 1b (not 4b): {len(features_in_270m_1b)}")
print(f"Features in 270m & 4b (not 1b): {len(features_in_270m_4b)}")
print(f"Features in 1b & 4b (not 270m): {len(features_in_1b_4b)}")
print()

print("=" * 80)
print("INSIGHT 2: PEAK STEP TIMING DISTRIBUTIONS")
print("=" * 80)

for model_name in model_order:
    candidates = models[model_name]['candidates']
    peak_steps = [c['peak_step'] for c in candidates]
    peak_step_counts = Counter(peak_steps)
    
    print(f"\n{model_name}:")
    print(f"  Total candidates: {len(candidates)}")
    print(f"  Peak step distribution:")
    for step in sorted(peak_step_counts.keys()):
        count = peak_step_counts[step]
        pct = 100 * count / len(candidates)
        print(f"    Step {step:2d}: {count:4d} ({pct:5.1f}%)")

print()

print("=" * 80)
print("INSIGHT 3: SUSTAIN RATIO STATISTICS (PLANNING FEATURES)")
print("=" * 80)

for model_name in model_order:
    candidates = models[model_name]['candidates']
    sustain_ratios = [c['sustain_ratio'] for c in candidates]
    
    avg_sustain = sum(sustain_ratios) / len(sustain_ratios) if sustain_ratios else 0
    min_sustain = min(sustain_ratios) if sustain_ratios else 0
    max_sustain = max(sustain_ratios) if sustain_ratios else 0
    
    print(f"{model_name}:")
    print(f"  Average sustain ratio: {avg_sustain:.4f}")
    print(f"  Min sustain ratio: {min_sustain:.4f}")
    print(f"  Max sustain ratio: {max_sustain:.4f}")
    print()

print("=" * 80)
print("INSIGHT 4: PERCENTILE RANKING CONSISTENCY")
print("=" * 80)

top_n = 10

for model_name in model_order:
    candidates = models[model_name]['candidates']
    top_candidates = sorted(candidates, key=lambda x: -x['peak_val'])[:top_n]
    print(f"\nTop {top_n} candidates in {model_name}:")
    for i, cand in enumerate(top_candidates, 1):
        print(f"  {i:2d}. L{cand['layer']:2d} F{cand['feat']:5d}  peak_val={cand['peak_val']:.6f}  rhyme_percentile={cand['rhyme_percentile']:.1f}%")

print()

print("=" * 80)
print("INSIGHT 5: SCALING LAWS - PLANNING VS EXECUTION FEATURES")
print("=" * 80)

for model_name in model_order:
    stats = models[model_name]['statistics']
    n_planning = stats['n_planning_features']
    n_execution = stats['n_execution_features']
    total = n_planning + n_execution
    planning_pct = 100 * n_planning / total if total > 0 else 0
    execution_pct = 100 * n_execution / total if total > 0 else 0
    
    print(f"{model_name}:")
    print(f"  Planning features: {n_planning:6d} ({planning_pct:5.1f}%)")
    print(f"  Execution features: {n_execution:6d} ({execution_pct:5.1f}%)")
    print()

print()

print("=" * 80)
print("INSIGHT 6: TEMPORAL CLUSTERING DISTRIBUTION")
print("=" * 80)

rhyme_steps = {name: models[name]['config']['rhyme_step'] for name in model_order}

for model_name in model_order:
    candidates = models[model_name]['candidates']
    rhyme_step = rhyme_steps[model_name]
    early_cutoff = rhyme_step // 3
    mid_cutoff = (2 * rhyme_step) // 3
    
    early_count = sum(1 for c in candidates if c['peak_step'] <= early_cutoff)
    mid_count = sum(1 for c in candidates if early_cutoff < c['peak_step'] <= mid_cutoff)
    late_count = sum(1 for c in candidates if c['peak_step'] > mid_cutoff)
    
    total = len(candidates)
    early_pct = 100 * early_count / total if total > 0 else 0
    mid_pct = 100 * mid_count / total if total > 0 else 0
    late_pct = 100 * late_count / total if total > 0 else 0
    
    print(f"{model_name} (rhyme_step={rhyme_step}):")
    print(f"  Early (0-{early_cutoff}):      {early_count:4d} ({early_pct:5.1f}%)")
    print(f"  Mid ({early_cutoff+1}-{mid_cutoff}):       {mid_count:4d} ({mid_pct:5.1f}%)")
    print(f"  Late ({mid_cutoff+1}-{rhyme_step}):     {late_count:4d} ({late_pct:5.1f}%)")
    print()

print()

print("=" * 80)
print("INSIGHT 7: RHYME-CIRCUIT CANDIDATE GROWTH AND EARLY SPIKE PREVALENCE")
print("=" * 80)

model_sizes = [270, 1024, 4096]
candidate_counts = []
early_spike_counts = []
early_spike_prevalence = []
unique_feature_counts = []

for model_name in model_order:
    stats = models[model_name]['statistics']
    candidates = models[model_name]['candidates']
    early_spikes = models[model_name]['early_spikes']
    
    n_candidates = stats['n_candidates']
    n_early_spikes = stats['n_early_spikes']
    n_unique = stats['n_unique_features']
    
    early_spike_pct = 100 * n_early_spikes / n_unique if n_unique > 0 else 0
    
    candidate_counts.append(n_candidates)
    early_spike_counts.append(n_early_spikes)
    early_spike_prevalence.append(early_spike_pct)
    unique_feature_counts.append(n_unique)
    
    print(f"{model_name}:")
    print(f"  Unique features above threshold: {n_unique:5d}")
    print(f"  Rhyme-circuit candidates: {n_candidates:5d}")
    print(f"  Early spike features: {n_early_spikes:5d} ({early_spike_pct:.1f}% of unique)")
    print()

print("Scaling analysis:")
for i in range(len(model_order) - 1):
    size_ratio = model_sizes[i+1] / model_sizes[i]
    cand_ratio = candidate_counts[i+1] / candidate_counts[i] if candidate_counts[i] > 0 else 0
    unique_ratio = unique_feature_counts[i+1] / unique_feature_counts[i] if unique_feature_counts[i] > 0 else 0
    
    print(f"\n{model_order[i]} → {model_order[i+1]}:")
    print(f"  Model size ratio: {size_ratio:.2f}x")
    print(f"  Candidate count ratio: {cand_ratio:.2f}x")
    print(f"  Unique features ratio: {unique_ratio:.2f}x")

print()

print("=" * 80)
print("ADDITIONAL: FEATURE INDEX MAPPING (SHARED FEATURES)")
print("=" * 80)

shared_all = features_candidates['270m'] & features_candidates['1b'] & features_candidates['4b']

if shared_all:
    print(f"\nShared features across all models: {len(shared_all)}")
    
    shared_list = sorted(list(shared_all))[:20]
    
    print("\nTop 20 shared features by appearance order:")
    for layer, feat in shared_list:
        rank_270m = next((i for i, c in enumerate(models['270m']['candidates']) if c['layer'] == layer and c['feat'] == feat), None)
        rank_1b = next((i for i, c in enumerate(models['1b']['candidates']) if c['layer'] == layer and c['feat'] == feat), None)
        rank_4b = next((i for i, c in enumerate(models['4b']['candidates']) if c['layer'] == layer and c['feat'] == feat), None)
        
        print(f"  L{layer:2d} F{feat:5d}  ranks: 270m={rank_270m}, 1b={rank_1b}, 4b={rank_4b}")
else:
    print("No features shared across all three models.")

print()
print("=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

print(f"\n{'Model':<10} {'Unique':<10} {'Candidates':<12} {'Early Spikes':<15} {'Planning %':<12} {'Execution %':<12}")
print("-" * 70)

for model_name in model_order:
    stats = models[model_name]['statistics']
    n_unique = stats['n_unique_features']
    n_cand = stats['n_candidates']
    n_early = stats['n_early_spikes']
    n_plan = stats['n_planning_features']
    n_exec = stats['n_execution_features']
    
    total = n_plan + n_exec
    plan_pct = 100 * n_plan / total if total > 0 else 0
    exec_pct = 100 * n_exec / total if total > 0 else 0
    
    print(f"{model_name:<10} {n_unique:<10d} {n_cand:<12d} {n_early:<15d} {plan_pct:<12.1f} {exec_pct:<12.1f}")