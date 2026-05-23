import torch
import torch.nn.functional as F
from collections import defaultdict
import json  # Fixed: Added missing import

def measure_downstream_effects_single_feature(
    model, prompt, layer, feat, suppression_step, device, tokenizer, RHYME_TOKEN, max_new_tokens=20, measurement_prompt=None 
):
    measurement_prompt = measurement_prompt or prompt
    inputs = model.tokenizer(measurement_prompt, return_tensors="pt").to(device)

    rhyme_token_id = tokenizer.encode(RHYME_TOKEN, add_special_tokens=False)[0]
    input_ids = inputs["input_ids"]  # shape [1, seq_len] — keep as 2D

    tl_model = model  # ReplacementModel IS the HookedTransformer

    # Baseline logits
    with torch.no_grad():
        logits_full = tl_model(input_ids)        # [1, seq, vocab]
        logits_original = logits_full[0, -1, :].unsqueeze(0)  # [1, vocab]
    probs_original = F.softmax(logits_original, dim=-1)

    # Confirm text change
    intervention = [(layer, suppression_step, feat, 0.0)]
    hook_result = model._get_feature_intervention_hooks(input_ids, intervention)
    hooks = hook_result[0]

    with torch.no_grad():
        logits_full_sup = model.run_with_hooks(
            input_ids,
            fwd_hooks=hooks
        )
        logits_suppressed = logits_full_sup[0, -1, :].unsqueeze(0)
    probs_suppressed = F.softmax(logits_suppressed, dim=-1)

    # Confirm text change (reuse same intervention)
    generated_suppressed = model.feature_intervention_generate(
        prompt, intervention, max_new_tokens=max_new_tokens, do_sample=False
    )[0]

    # Probability and Rank Analysis
    original_prob = probs_original[0, rhyme_token_id].item()
    suppressed_prob = probs_suppressed[0, rhyme_token_id].item()
    prob_drop = original_prob - suppressed_prob

    original_sorted_probs, original_sorted_indices = torch.sort(probs_original[0], descending=True)
    original_rank = (original_sorted_indices == rhyme_token_id).nonzero(as_tuple=True)[0].item()

    suppressed_sorted_probs, suppressed_sorted_indices = torch.sort(probs_suppressed[0], descending=True)
    suppressed_rank = (suppressed_sorted_indices == rhyme_token_id).nonzero(as_tuple=True)[0].item()

    rank_shift = original_rank - suppressed_rank

    # Top Token Decoding
    original_top5_indices = original_sorted_indices[:5].tolist()
    original_top5_tokens = [tokenizer.decode([idx]) for idx in original_top5_indices]
    original_top5_probs = original_sorted_probs[:5].tolist()

    suppressed_top5_indices = suppressed_sorted_indices[:5].tolist()
    suppressed_top5_tokens = [tokenizer.decode([idx]) for idx in suppressed_top5_indices]
    suppressed_top5_probs = suppressed_sorted_probs[:5].tolist()

    # Entropy Calculations
    original_entropy = -(probs_original[0] * torch.log(probs_original[0] + 1e-10)).sum().item()
    suppressed_entropy = -(probs_suppressed[0] * torch.log(probs_suppressed[0] + 1e-10)).sum().item()
    entropy_increase = suppressed_entropy - original_entropy

    return {
        "layer": layer,
        "feat": feat,
        "suppression_step": suppression_step,
        "original_prob": original_prob,
        "suppressed_prob": suppressed_prob,
        "prob_drop": prob_drop,
        "prob_drop_pct": 100 * prob_drop / original_prob if original_prob > 0 else 0,
        "original_rank": original_rank,
        "suppressed_rank": suppressed_rank,
        "rank_shift": rank_shift,
        "original_top5_tokens": original_top5_tokens,
        "original_top5_probs": original_top5_probs,
        "suppressed_top5_tokens": suppressed_top5_tokens,
        "suppressed_top5_probs": suppressed_top5_probs,
        "original_entropy": original_entropy,
        "suppressed_entropy": suppressed_entropy,
        "entropy_increase": entropy_increase,
        "generated_suppressed": generated_suppressed,
    }


def measure_downstream_effects_batch(model, prompt, candidates_list, suppression_step_key, device, tokenizer, RHYME_TOKEN, max_new_tokens=20, max_candidates=None, measurement_prompt=None):
    """
    Measure downstream effects for multiple candidates.
    """
    results = []
    candidates_to_test = candidates_list[:max_candidates] if max_candidates else candidates_list
    
    for i, candidate in enumerate(candidates_to_test):
        layer = candidate["feat_key"][0]
        feat = candidate["feat_key"][1]
        
        if suppression_step_key == 'peak_step':
            suppression_step = candidate['peak_step']
        elif suppression_step_key == 'first_step':
            suppression_step = candidate['first_step']
        else:
            suppression_step = suppression_step_key
        
        try:
            effect = measure_downstream_effects_single_feature(
                model, prompt, layer, feat, suppression_step, device, tokenizer, RHYME_TOKEN, max_new_tokens, measurement_prompt=measurement_prompt
            )
            results.append(effect)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1} / {len(candidates_to_test)} candidates")
        
        except Exception as e:
            # Fixed: Provide more detail on errors during batch processing
            print(f"  Error processing L{layer} F{feat}: {type(e).__name__} - {e}")
            continue
    
    return results


def analyze_downstream_effects(results, top_n=20):
    """
    Analyze results. Returns a safe dictionary even if results are empty.
    """
    if not results:
        print("No results to analyze.")
        # Fixed: Return a default structure so subscripting doesn't crash the main script
        return {
            'sorted_by_prob_drop': [],
            'sorted_by_rank_shift': [],
            'sorted_by_entropy_increase': [],
            'aggregate': {
                'avg_prob_drop': 0, 'avg_prob_drop_pct': 0, 
                'avg_rank_shift': 0, 'avg_entropy_increase': 0, 
                'broken_rhyme_count': 0
            }
        }
    
    sorted_by_prob_drop = sorted(results, key=lambda x: -x['prob_drop'])
    sorted_by_rank_shift = sorted(results, key=lambda x: -x['rank_shift'])
    sorted_by_entropy_increase = sorted(results, key=lambda x: -x['entropy_increase'])
    
    # (Print statements omitted for brevity, keeping existing logic)
    
    avg_prob_drop = sum(r['prob_drop'] for r in results) / len(results)
    avg_prob_drop_pct = sum(r['prob_drop_pct'] for r in results) / len(results)
    avg_rank_shift = sum(r['rank_shift'] for r in results) / len(results)
    avg_entropy_increase = sum(r['entropy_increase'] for r in results) / len(results)
    broken_rhyme = sum(1 for r in results if r['suppressed_rank'] > 100)
    
    return {
        'sorted_by_prob_drop': sorted_by_prob_drop,
        'sorted_by_rank_shift': sorted_by_rank_shift,
        'sorted_by_entropy_increase': sorted_by_entropy_increase,
        'aggregate': {
            'avg_prob_drop': avg_prob_drop,
            'avg_prob_drop_pct': avg_prob_drop_pct,
            'avg_rank_shift': avg_rank_shift,
            'avg_entropy_increase': avg_entropy_increase,
            'broken_rhyme_count': broken_rhyme,
        }
    }


def save_downstream_effects_to_json(results, analyzed, output_filename):
    """
    Save downstream effects results to JSON.
    """
    # Defensive check for analyzed dictionary structure
    top_drop = analyzed.get('sorted_by_prob_drop', [])
    top_rank = analyzed.get('sorted_by_rank_shift', [])
    top_entropy = analyzed.get('sorted_by_entropy_increase', [])

    output = {
        "downstream_effects": results,
        "analysis": {
            "top_by_prob_drop": [
                {
                    "layer": r["layer"], "feat": r["feat"],
                    "suppression_step": r["suppression_step"],
                    "prob_drop": float(r["prob_drop"]),
                    "prob_drop_pct": float(r["prob_drop_pct"]),
                    "rank_shift": int(r["rank_shift"]),
                } for r in top_drop[:50]
            ],
            "top_by_rank_shift": [
                {
                    "layer": r["layer"], "feat": r["feat"],
                    "suppression_step": r["suppression_step"],
                    "rank_shift": int(r["rank_shift"]),
                    "prob_drop_pct": float(r["prob_drop_pct"]),
                } for r in top_rank[:50]
            ],
            "top_by_entropy_increase": [
                {
                    "layer": r["layer"], "feat": r["feat"],
                    "suppression_step": r["suppression_step"],
                    "entropy_increase": float(r["entropy_increase"]),
                    "prob_drop_pct": float(r["prob_drop_pct"]),
                } for r in top_entropy[:50]
            ],
            "aggregate_stats": analyzed.get("aggregate", {}),
        }
    }
    
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDownstream effects saved to {output_filename}")