"""Suppression measurement: what happens to P(rhyme token) when a feature is zeroed.

Two-phase by design.

  * `measure_features()` is the cheap phase -- one forward pass per feature, no
    generation. The unsuppressed baseline is computed **once** and reused, since
    it does not depend on which feature is being suppressed.
  * `generate_for()` is the expensive phase -- ~20 sequential forwards per call.
    It runs only for the subset selected from phase one's ranking.

Splitting them is a correctness fix as much as a performance one. When the two
shared a `try` block, an OOM inside generation discarded the already-computed
probability/rank/entropy row for that feature: the append never ran. Each phase
now has its own error isolation, so a generation failure costs a generation.

Positions, not steps
--------------------
`position` in every row here is a **token index into the measurement sequence**,
which is what `_get_feature_intervention_hooks` indexes with. It is *not* a
generation step number. `tracing.step_contexts()` does the conversion; nothing
in this module infers a position.

Logits, not just probabilities
------------------------------
Rows carry `original_logit` / `suppressed_logit` for the rhyme token alongside
the full-vocabulary `logsumexp` of each pass. Probabilities alone cannot express
the cross-pass contrast -- the normaliser differs between the two passes -- and
`prob_drop` is bounded above by `original_prob`, so it saturates exactly where
the rhyme is already unlikely. See `methodology_evidence.md` §1 (Zhang & Nanda).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

TOP_K_RECORDED = 10


def candidate_layer_feat(candidate: dict) -> tuple[int, int]:
    """(layer, feat) from either candidate shape.

    Callers pass one of two dict layouts: flat {"layer", "feat"} (what tracing.py
    builds, and what it writes into the results JSON) or {"feat_key": (layer,
    feat)} (the internal timeline representation). Reading only one of them
    raises KeyError above the per-candidate try/except, which kills a whole
    intervention pass rather than skipping one feature.
    """
    if "feat_key" in candidate:
        return int(candidate["feat_key"][0]), int(candidate["feat_key"][1])
    return int(candidate["layer"]), int(candidate["feat"])


def rhyme_token_id(tokenizer, rhyme_token: str) -> int:
    ids = tokenizer.encode(rhyme_token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"rhyme token {rhyme_token!r} encodes to {len(ids)} tokens {ids}; "
            "single-token rhymes only -- callers must check label['single_token']"
        )
    return int(ids[0])


def _summarize_pass(logits_1d: torch.Tensor, token_id: int) -> dict:
    """Everything read off one next-token logit vector.

    `logsumexp` is taken over the **full vocabulary**, before any top-k slice.
    Computed over the top-k slice instead it would not be the softmax
    denominator, and `exp(logit - logsumexp)` would silently disagree with the
    `prob` stored beside it.
    """
    logits = logits_1d.float()
    lse = torch.logsumexp(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    rank = int((sorted_idx == token_id).nonzero(as_tuple=True)[0].item())

    top_idx = sorted_idx[:TOP_K_RECORDED]
    return {
        "logit": float(logits[token_id].item()),
        "logsumexp": float(lse.item()),
        "prob": float(probs[token_id].item()),
        "rank": rank,
        "entropy": float(-(probs * torch.log(probs + 1e-10)).sum().item()),
        "top_ids": [int(i) for i in top_idx.tolist()],
        "top_logits": [float(v) for v in logits[top_idx].tolist()],
        "top_probs": [float(v) for v in sorted_probs[:TOP_K_RECORDED].tolist()],
    }


def compute_baseline(model, tokens: torch.Tensor, token_id: int) -> dict:
    """Unsuppressed next-token distribution for the measurement sequence.

    Hoisted out of the per-feature loop: it is identical for every feature, and
    was previously recomputed once per candidate.
    """
    input_ids = tokens.unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids)[0, -1, :]
    return _summarize_pass(logits, token_id)


def measure_features(
    model,
    tokens: torch.Tensor,
    baseline: dict,
    rows: list[dict],
    token_id: int,
    progress_every: int = 25,
) -> tuple[list[dict], list[dict]]:
    """Phase one. Suppress each row's feature at its own position; read the effect.

    `rows` carry `layer`, `feat`, `position`, and whatever provenance the caller
    wants echoed back (`populations`, `step_keys`, ...). Returns
    (measured, failures) -- failures are recorded rather than dropped, so a
    shortfall in coverage has a stated cause instead of being invisible.
    """
    input_ids = tokens.unsqueeze(0)
    n_pos = int(tokens.shape[0])

    measured: list[dict] = []
    failures: list[dict] = []

    for i, row in enumerate(rows):
        layer, feat = candidate_layer_feat(row)
        position = int(row["position"])

        # Raise, do not log: the per-feature except below would otherwise absorb
        # an out-of-range position into a silently dropped row. A position past
        # the end of the measurement sequence is a bug in position derivation,
        # not a feature that happens not to work.
        if not 0 <= position < n_pos:
            raise IndexError(
                f"suppression position {position} outside measurement sequence "
                f"of length {n_pos} (L{layer} F{feat}). Positions must be derived "
                "from tracing.step_contexts() against this same prompt."
            )

        try:
            hooks = model._get_feature_intervention_hooks(
                input_ids, [(layer, position, feat, 0.0)]
            )[0]
            with torch.no_grad():
                logits = model.run_with_hooks(input_ids, fwd_hooks=hooks)[0, -1, :]
            sup = _summarize_pass(logits, token_id)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            failures.append(
                {
                    "layer": layer,
                    "feat": feat,
                    "position": position,
                    "cause": "measurement_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  measurement failed L{layer} F{feat} @pos{position}: {type(exc).__name__}")
            continue

        out = dict(row)
        out.update(
            {
                "layer": layer,
                "feat": feat,
                "position": position,
                "original_logit": baseline["logit"],
                "suppressed_logit": sup["logit"],
                "logit_drop": baseline["logit"] - sup["logit"],
                "original_logsumexp": baseline["logsumexp"],
                "suppressed_logsumexp": sup["logsumexp"],
                "original_prob": baseline["prob"],
                "suppressed_prob": sup["prob"],
                "prob_drop": baseline["prob"] - sup["prob"],
                "prob_drop_pct": (
                    100 * (baseline["prob"] - sup["prob"]) / baseline["prob"]
                    if baseline["prob"] > 0
                    else 0.0
                ),
                "original_rank": baseline["rank"],
                "suppressed_rank": sup["rank"],
                "rank_shift": baseline["rank"] - sup["rank"],
                "original_entropy": baseline["entropy"],
                "suppressed_entropy": sup["entropy"],
                "entropy_increase": sup["entropy"] - baseline["entropy"],
                "original_top": {
                    "ids": baseline["top_ids"],
                    "logits": baseline["top_logits"],
                    "probs": baseline["top_probs"],
                },
                "suppressed_top": {
                    "ids": sup["top_ids"],
                    "logits": sup["top_logits"],
                    "probs": sup["top_probs"],
                },
            }
        )
        measured.append(out)

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  measured {i + 1} / {len(rows)}")

    return measured, failures


def generate_for(model, context_prompt, interventions, max_new_tokens: int = 20) -> str | None:
    """Phase two. One greedy continuation, isolated from phase one's errors.

    `context_prompt` must be the step context the feature's position indexes
    into -- integer positions do not survive into generated tokens
    (`_convert_open_ended_interventions`), so generating from a shorter prompt
    would put the position out of range.
    """
    try:
        return model.feature_intervention_generate(
            context_prompt, interventions, max_new_tokens=max_new_tokens, do_sample=False
        )[0]
    except Exception as exc:  # noqa: BLE001
        print(f"  generation failed ({type(exc).__name__}: {exc})")
        return None


def analyze_downstream_effects(results: list[dict]) -> dict:
    """Rankings and aggregate stats. Safe on an empty list."""
    if not results:
        return {
            "sorted_by_logit_drop": [],
            "sorted_by_prob_drop": [],
            "sorted_by_rank_shift": [],
            "sorted_by_entropy_increase": [],
            "aggregate": {
                "n": 0,
                "avg_logit_drop": None,
                "avg_prob_drop": None,
                "avg_prob_drop_pct": None,
                "avg_rank_shift": None,
                "avg_entropy_increase": None,
                "broken_rhyme_count": 0,
            },
        }

    n = len(results)
    return {
        "sorted_by_logit_drop": sorted(results, key=lambda x: -x["logit_drop"]),
        "sorted_by_prob_drop": sorted(results, key=lambda x: -x["prob_drop"]),
        "sorted_by_rank_shift": sorted(results, key=lambda x: -x["rank_shift"]),
        "sorted_by_entropy_increase": sorted(results, key=lambda x: -x["entropy_increase"]),
        "aggregate": {
            "n": n,
            "avg_logit_drop": sum(r["logit_drop"] for r in results) / n,
            "avg_prob_drop": sum(r["prob_drop"] for r in results) / n,
            "avg_prob_drop_pct": sum(r["prob_drop_pct"] for r in results) / n,
            "avg_rank_shift": sum(r["rank_shift"] for r in results) / n,
            "avg_entropy_increase": sum(r["entropy_increase"] for r in results) / n,
            "broken_rhyme_count": sum(1 for r in results if r["suppressed_rank"] > 100),
        },
    }
