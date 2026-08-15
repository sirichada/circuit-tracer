"""Shared tracing pipeline for all Gemma-3 sizes.

Consolidates what used to be ~730 near-identical lines in each of
`tracing-{270m,1b,4b}.py`. Those three now supply only their config and call
`run()` here. Keeping one copy is what stops `RHYME_TOKEN` / `GRAPH_DIR` /
the checkpoint names from going stale in three places at once.

Two halves:

  * `analyze_prompt()` and everything it calls is **pure** -- graph JSON in,
    statistics out. No model, no GPU. Run it anywhere.
  * `run_interventions()` needs the model on a GPU.

Per-prompt rhyme targets come from `rhyme_labels.json` (produced by
`rhyme_labels.py`), never from hardcoded constants.

    python experiment/tracing-1b.py                    # analysis + interventions
    python experiment/tracing-1b.py --no-interventions # GPU-free half only
    python experiment/tracing-1b.py --slugs realm ten  # selected prompts
"""

from __future__ import annotations

import json
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
EARLY_SPIKE_PERCENTILE = 70.0
MAX_CANDIDATES_MEASURED = 50
TOP_N_REPORTED = 15

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
    """(layer, local_feat) from a node dict, handling both id formats."""
    m = re.match(r"^(\d+)_(\d+)-", node.get("jsNodeId", ""))
    if m:
        return int(m.group(1)), int(m.group(2))
    layer, feat = node.get("layer"), node.get("feature")
    if layer is not None and feat is not None:
        return int(layer), int(feat)
    return None, None


def load_step_features(slug_dir: Path) -> tuple[dict, dict, dict]:
    """Read every step-*.json, keeping transcoder nodes above the influence
    threshold. Returns (per-step rows, per-step total |influence|, step tokens).
    """
    step_features: dict[int, list[dict]] = {}
    step_total: dict[int, float] = {}
    step_tokens: dict[int, str] = {}

    for path in sorted(slug_dir.glob("step-*.json")):
        m = STEP_RE.search(path.name)
        if not m:
            continue
        step_idx = int(m.group(1))
        data = json.loads(path.read_text())

        rows, total = [], 0.0
        for node in data.get("nodes", []):
            if "transcoder" not in node.get("feature_type", ""):
                continue
            inf = node.get("influence") or 0
            if inf == 0 or abs(inf) < INFLUENCE_THRESHOLD:
                continue
            layer, feat = parse_node_ids(node)
            if layer is None:
                continue
            rows.append({"layer": layer, "feat": feat, "influence": abs(inf), "raw_inf": inf})
            total += abs(inf)

        step_features[step_idx] = rows
        step_total[step_idx] = total
        step_tokens[step_idx] = m.group(2).replace("_", " ")

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


def feature_stats(timeline: dict, percentiles: dict, rhyme_step: int) -> list[dict]:
    stats = []
    for key, step_inf in timeline.items():
        if not step_inf:
            continue
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
                "rhyme_val": rhyme_val,
                "rhyme_percentile": percentiles[key].get(rhyme_step, 0.0),
                "sustain_ratio": rhyme_val / peak_val if peak_val > 0 else 0.0,
            }
        )
    return stats


def select_candidates(planning: list[dict], rhyme_step_features: set) -> list[dict]:
    """Planning features that are also prominent *at* the rhyme step and hold
    a good share of their peak influence until then."""
    out = [
        e
        for e in planning
        if e["feat_key"] in rhyme_step_features
        and e["rhyme_percentile"] >= CANDIDATE_MIN_RHYME_PERCENTILE
        and e["sustain_ratio"] >= CANDIDATE_MIN_SUSTAIN
    ]
    out.sort(key=lambda x: -(x["peak_val"] + x["rhyme_val"]))
    return out


def find_early_spikes(percentiles: dict, stats_by_key: dict) -> list[dict]:
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
    ) + extra
    out = []
    for e in entries:
        row = {"layer": e["feat_key"][0], "feat": e["feat_key"][1]}
        for k in keys:
            if k in e:
                row[k] = float(e[k]) if isinstance(e[k], float) else e[k]
        out.append(row)
    return out


def analyze_prompt(cfg: SizeConfig, slug: str, label: dict, verbose: bool = True) -> dict[str, Any]:
    """Full GPU-free analysis for one prompt. Returns the results payload that
    `run_interventions()` later extends."""
    slug_dir = cfg.graph_dir / slug
    rhyme_step = label["rhyme_step"]

    step_features, step_total, step_tokens = load_step_features(slug_dir)
    if not step_features:
        raise ValueError(f"no graphs in {slug_dir}")

    timeline, percentiles = build_timeline(step_features, step_total)
    stats = feature_stats(timeline, percentiles, rhyme_step)
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

    if verbose:
        print(
            f"\n{'=' * 70}\n[{cfg.size}/{slug}] {label['label'].upper()}  "
            f"{label['rhyme_word']} vs {label['target_word']} @ step {rhyme_step}\n{'=' * 70}"
        )
        print(
            f"  steps={len(step_features)}  features={len(timeline)}  "
            f"planning={len(planning)}  execution={len(execution)}  candidates={len(candidates)}"
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
            "early_cutoff": rhyme_step // 3,
            "mid_cutoff": (2 * rhyme_step) // 3,
        },
        "statistics": {
            "n_steps": len(step_features),
            "n_unique_features": len(timeline),
            "n_planning_features": len(planning),
            "n_execution_features": len(execution),
            "n_candidates": len(candidates),
            "n_early_spikes": len(spikes),
            "band_counts": {k: len(v) for k, v in bands.items()},
        },
        "candidates": _serialize(candidates),
        "early_spikes": _serialize(spikes, extra=("early_spike_step",)),
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


def load_model(cfg: SizeConfig):
    """Load the -it model and its matching Gemma-Scope-2 transcoders."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    from circuit_tracer import ReplacementModel
    from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set

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
        dtype=torch.bfloat16,
        device=device,
    )
    return model, tokenizer, device


def run_interventions(model, tokenizer, device, cfg, slug, label, record, results) -> dict:
    """Suppress each candidate feature and measure the drop in P(rhyme token).

    Runs the same measurement at two suppression points -- the feature's peak
    step and its first active step -- as the original scripts did.
    """
    from downstream_effects_addon import (
        analyze_downstream_effects,
        measure_downstream_effects_batch,
    )

    if not label["single_token"]:
        print(
            f"  [{slug}] SKIPPING interventions: rhyme token {label['rhyme_token']!r} is "
            f"multi-token, so encode(...)[0] would measure the wrong token"
        )
        results["interventions"] = {"skipped": "multi-token rhyme word"}
        return results

    prompt = record["prompt_text"]
    measurement_prompt = prompt + label["prefix_before_rhyme"]
    candidates = [
        {
            "layer": c["layer"],
            "feat": c["feat"],
            "peak_step": c["peak_step"],
            "first_step": c["first_step"],
        }
        for c in results["candidates"]
    ]
    if not candidates:
        results["interventions"] = {"skipped": "no candidates"}
        return results

    baseline = model.feature_intervention_generate(prompt, [], do_sample=False)[0]
    print(f"  baseline: {baseline}")

    section = {"baseline_output": baseline, "measurement_prompt": measurement_prompt}
    for key in ("peak_step", "first_step"):
        raw = measure_downstream_effects_batch(
            model=model,
            prompt=prompt,
            candidates_list=candidates,
            suppression_step_key=key,
            device=device,
            tokenizer=tokenizer,
            RHYME_TOKEN=label["rhyme_token"],
            max_new_tokens=20,
            max_candidates=MAX_CANDIDATES_MEASURED,
            measurement_prompt=measurement_prompt,
        )
        analyzed = analyze_downstream_effects(raw, top_n=TOP_N_REPORTED)
        # Serialize every measured result. The old scripts sliced [:30] while
        # measuring 50, which silently turned any mean over this set into a
        # truncated mean.
        section[key] = {
            "n_measured": len(raw),
            "results": raw,
            "summary": {k: v for k, v in analyzed.items() if k != "sorted_by_prob_drop"},
        }
        for r in sorted(raw, key=lambda x: -x["prob_drop"])[:5]:
            out = model.feature_intervention_generate(
                prompt,
                [(r["layer"], r["suppression_step"], r["feat"], 0.0)],
                max_new_tokens=20,
                do_sample=False,  # must match the greedy baseline above
            )[0]
            r["post_intervention_output"] = out

    results["interventions"] = section
    return results


# ------------------------------------------------------------------------ entry


def run(cfg: SizeConfig, slugs: list[str] | None = None, interventions: bool = True) -> None:
    if not LABELS_PATH.exists():
        raise SystemExit(f"{LABELS_PATH} missing -- run `python experiment/rhyme_labels.py` first")

    labels = {(r["size"], r["slug"]): r for r in json.loads(LABELS_PATH.read_text())}
    prompt_set = {r["slug"]: r for r in json.loads(PROMPT_SET_PATH.read_text())}

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
                model, tokenizer, device = load_model(cfg)
            results = run_interventions(
                model, tokenizer, device, cfg, slug, label, prompt_set[slug], results
            )

        out = cfg.results_path(slug)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"  wrote {out.relative_to(EXPERIMENT)}")


def main(size: str) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=f"Circuit tracing for Gemma-3-{size}-it")
    ap.add_argument("--slugs", nargs="*", help="prompt slugs (default: all with graphs)")
    ap.add_argument(
        "--no-interventions", action="store_true", help="run the GPU-free analysis half only"
    )
    args = ap.parse_args()

    run(CONFIGS[size], slugs=args.slugs, interventions=not args.no_interventions)
