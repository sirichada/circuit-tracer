"""Threshold sensitivity and feature-selection controls. GPU-free.

Answers the reviewer question behind `action_plan.md` Phase 4 step 9 -- "are the
selected features genuinely rhyme-specific, or just generically high-influence?"
-- in three parts:

  1. **Sweep**: re-select candidates under a grid of cutoffs and report how
     `n_candidates` and the top suppression effects move.
  2. **Near-miss**: features that fail the candidate filter by a small margin.
     If they move the rhyme probability, the cutoff is discarding real signal.
  3. **Random control**: non-candidate features matched to the candidate set's
     influence distribution. If they move it just as much, the selection isn't
     distinguishing anything.

**This script no longer loads a model.** It used to run its own interventions
for parts 2 and 3, which meant a second model load per size and a measurement
set that could silently disagree with tracing's. `tracing.py` now measures the
union of all four populations in one pass and tags each row with its
memberships; everything here is a join against `circuit_tracing_results_*.json`.

Coverage has three distinguishable causes
-----------------------------------------
`run_sweep` recomputes `feature_stats` at every influence floor, and `peak_step`
is an argmax over *normalised* influence -- the normaliser changes with the
floor, so a feature's `peak_step` at one grid cell need not be the one the GPU
pass measured. That makes `n_measured < n_candidates` ambiguous between three
quite different things, and the previous version reported a single number that
conflated them. Every sweep row now breaks the shortfall down:

  * `hit`                       -- measured at exactly this cell's position
  * `measured_at_other_position` -- this feature was measured, but its peak_step
                                    moved with the floor, so no row exists here
  * `not_measured`              -- never entered the measurement set at all
  * `measurement_failed`        -- dispatched and errored (see `failures`)

Only the last is a defect. The second is a property of the sweep and is the
reason `peak_step`-vs-`first_step` stability is worth reporting.

    python experiment/threshold_sensitivity.py --size 4b
    python experiment/threshold_sensitivity.py --size 4b --slugs realm ten
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from tracing import (
    CANDIDATE_MIN_RHYME_PERCENTILE,
    CANDIDATE_MIN_SUSTAIN,
    CONFIGS,
    INFLUENCE_GRID,
    INFLUENCE_THRESHOLD,
    LABELS_PATH,
    PERCENTILE_GRID,
    PERCENTILE_MARGIN,
    RANDOM_SEED,
    STEP_KEYS,
    SUSTAIN_GRID,
    SUSTAIN_MARGIN,
    TOP_N_REPORTED,
    SizeConfig,
    build_timeline,
    candidate_keys,
    feature_stats,
    filter_at,
    load_raw_step_nodes,
    sample_matched_control,
    split_populations,
    step_contexts,
)

OUT_DIR = Path(__file__).parent / "threshold"

TOP_K_SWEEP = 5
# Headline suppression statistic. A mean over "all candidates" is not comparable
# across prompts -- the population size scales with continuation length, and a
# max is a max over a bigger draw. Fixed-k fixes the draw.
FIXED_K = 10


# ------------------------------------------------------------------ measured rows


class Measurements:
    """The GPU pass's output, indexed for joining.

    Positions are threshold-independent (they come from `step_contexts`), which
    is what makes a join on `(layer, feat, position)` possible at all: a cell's
    recomputed `peak_step` maps to a position without re-running anything.
    """

    def __init__(self, payload: dict | None):
        self.available = False
        self.by_position: dict[tuple[int, int, int], dict] = {}
        self.positions_by_feat: dict[tuple[int, int], set[int]] = {}
        self.failed: set[tuple[int, int, int]] = set()
        self.rows: list[dict] = []

        section = (payload or {}).get("interventions")
        if not isinstance(section, dict) or "results" not in section:
            return  # missing, never run, or {"skipped": "..."}

        self.available = True
        self.rows = section["results"]
        for r in self.rows:
            key = (r["layer"], r["feat"], r["position"])
            self.by_position[key] = r
            self.positions_by_feat.setdefault((r["layer"], r["feat"]), set()).add(r["position"])
        for f in section.get("failures", []):
            self.failed.add((f["layer"], f["feat"], f["position"]))

    def population(self, name: str) -> list[dict]:
        return [r for r in self.rows if name in r.get("populations", [])]

    def classify(self, layer: int, feat: int, position: int) -> tuple[str, dict | None]:
        key = (layer, feat, position)
        if key in self.by_position:
            return "hit", self.by_position[key]
        if key in self.failed:
            return "measurement_failed", None
        if (layer, feat) in self.positions_by_feat:
            return "measured_at_other_position", None
        return "not_measured", None


def load_measurements(cfg: SizeConfig, slug: str) -> Measurements:
    path = cfg.results_path(slug)
    payload = json.loads(path.read_text()) if path.exists() else None
    return Measurements(payload)


# ------------------------------------------------------------------- part 1: sweep


def run_sweep(
    step_nodes: dict,
    contexts: dict[int, dict],
    rhyme_step: int,
    measured: Measurements,
    n_layers: int,
) -> list[dict]:
    """One row per (influence, percentile, sustain) cell."""
    rows = []
    for influence in INFLUENCE_GRID:
        step_features, step_total = filter_at(step_nodes, influence)
        timeline, percentiles = build_timeline(step_features, step_total)
        stats = feature_stats(timeline, percentiles, rhyme_step, n_layers)
        rhyme_step_features = {(r["layer"], r["feat"]) for r in step_features.get(rhyme_step, [])}

        for min_pct in PERCENTILE_GRID:
            for min_sustain in SUSTAIN_GRID:
                cands = candidate_keys(stats, rhyme_step_features, rhyme_step, min_pct, min_sustain)
                row: dict[str, Any] = {
                    "influence_threshold": influence,
                    "min_rhyme_percentile": min_pct,
                    "min_sustain": min_sustain,
                    "n_candidates": len(cands),
                    "is_shipped_setting": (
                        influence == INFLUENCE_THRESHOLD
                        and min_pct == CANDIDATE_MIN_RHYME_PERCENTILE
                        and min_sustain == CANDIDATE_MIN_SUSTAIN
                    ),
                }
                if not measured.available:
                    row["coverage"] = None
                    row["n_measured"] = None
                    row["top_drops"] = None
                    row["fixed_k"] = None
                    rows.append(row)
                    continue

                causes: Counter = Counter()
                hits = []
                for c in cands:
                    layer, feat = c["feat_key"]
                    step = c["peak_step"]
                    if step not in contexts:
                        causes["no_step_context"] += 1
                        continue
                    cause, hit = measured.classify(layer, feat, contexts[step]["position"])
                    causes[cause] += 1
                    if hit is not None:
                        hits.append(hit)

                hits.sort(key=lambda r: -r["logit_drop"])
                row["coverage"] = dict(causes)
                row["n_measured"] = len(hits)
                # Top-K, not just the max: a maximum cannot move under a cutoff
                # that doesn't exclude the maximising feature, so reporting only
                # the max makes any threshold look more inert than it is.
                row["top_drops"] = [
                    {
                        "layer": r["layer"],
                        "feat": r["feat"],
                        "position": r["position"],
                        "logit_drop": r["logit_drop"],
                        "prob_drop": r["prob_drop"],
                        "prob_drop_pct": r["prob_drop_pct"],
                    }
                    for r in hits[:TOP_K_SWEEP]
                ]
                row["fixed_k"] = fixed_k_stats(hits, FIXED_K)
                rows.append(row)
    return rows


# --------------------------------------------------------------------- statistics


def best_per_feature(rows: list[dict]) -> list[dict]:
    """One row per (layer, feat): its strongest suppression across positions.

    A feature contributes up to two measured rows -- one at its `peak_step`
    position, one at its `first_step` position -- so counting rows would let a
    single feature occupy two slots in a top-k, and would compare row counts
    against feature counts. Every per-feature statistic collapses first.
    """
    best: dict[tuple[int, int], dict] = {}
    for r in rows:
        key = (r["layer"], r["feat"])
        if key not in best or r["logit_drop"] > best[key]["logit_drop"]:
            best[key] = r
    return list(best.values())


def fixed_k_stats(rows: list[dict], k: int) -> dict:
    """Mean over the top-k *features* by logit drop, or None if fewer than k.

    Reported instead of a mean over everything because population size scales
    with continuation length, and instead of a max because a max over a larger
    draw is larger for free. Prompts with fewer than k are excluded and said so.
    """
    per_feature = best_per_feature(rows)
    top = sorted(per_feature, key=lambda r: -r["logit_drop"])[:k]
    if len(top) < k:
        return {"k": k, "n_available": len(per_feature), "sufficient": False}
    return {
        "k": k,
        "n_available": len(per_feature),
        "sufficient": True,
        "mean_logit_drop": statistics.fmean(r["logit_drop"] for r in top),
        "mean_prob_drop": statistics.fmean(r["prob_drop"] for r in top),
        "mean_prob_drop_pct": statistics.fmean(r["prob_drop_pct"] for r in top),
    }


def summarize(rows: list[dict], n_selected: int) -> dict:
    """Row-level effect statistics plus the feature-level coverage check.

    `n_selected` counts *features* chosen by the analysis; `n_features_measured`
    counts features that came back from the GPU pass. With the measurement cap
    gone these must agree -- a gap now means a measurement genuinely failed,
    where it used to mean a limit was hit. `n_rows_measured` is larger than both
    whenever a feature was measured at two positions.
    """
    if not rows:
        return {
            "n_selected": n_selected,
            "n_features_measured": 0,
            "n_rows_measured": 0,
            "shortfall": n_selected,
            "fixed_k": fixed_k_stats([], FIXED_K),
        }
    logit = [r["logit_drop"] for r in rows]
    prob = [r["prob_drop"] for r in rows]
    n_feats = len({(r["layer"], r["feat"]) for r in rows})
    return {
        "n_selected": n_selected,
        "n_features_measured": n_feats,
        "n_rows_measured": len(rows),
        "shortfall": n_selected - n_feats,
        "mean_logit_drop": statistics.fmean(logit),
        "max_logit_drop": max(logit),
        "mean_prob_drop": statistics.fmean(prob),
        "max_prob_drop": max(prob),
        "n_positive_logit": sum(1 for d in logit if d > 0),
        "fixed_k": fixed_k_stats(rows, FIXED_K),
    }


def bootstrap_means(values: list[float], seed: int, n_resamples: int = 2000) -> list[float]:
    rng = random.Random(seed)
    return sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n_resamples))


def permutation_test(a: list[float], b: list[float], seed: int, n: int = 10000) -> float | None:
    """Two-sided p for mean(a) - mean(b) under random relabelling."""
    if len(a) < 2 or len(b) < 2:
        return None
    observed = abs(statistics.fmean(a) - statistics.fmean(b))
    pooled = a + b
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        rng.shuffle(pooled)
        diff = abs(statistics.fmean(pooled[: len(a)]) - statistics.fmean(pooled[len(a) :]))
        if diff >= observed:
            hits += 1
    return (hits + 1) / (n + 1)


def compare(groups: dict[str, list[dict]], seed: int) -> dict:
    """Candidates against the matched-random control, on the logit metric.

    The previous version located the candidate **mean** inside a distribution of
    random **individual** drops -- mismatched units, so the resulting percentile
    was not interpretable. Replaced with a permutation test on the difference of
    means, plus the bootstrap CI of the control mean for scale.
    """
    per_feature = {name: best_per_feature(rows) for name, rows in groups.items()}
    out: dict[str, Any] = {
        name: {
            "n_features": len(rows),
            "mean_logit_drop": statistics.fmean(r["logit_drop"] for r in rows) if rows else None,
            "max_logit_drop": max((r["logit_drop"] for r in rows), default=None),
            "mean_prob_drop": statistics.fmean(r["prob_drop"] for r in rows) if rows else None,
            "fixed_k": fixed_k_stats(rows, FIXED_K),
        }
        for name, rows in per_feature.items()
    }
    cand = [r["logit_drop"] for r in per_feature.get("candidate", [])]
    rand = [r["logit_drop"] for r in per_feature.get("random_control", [])]
    if cand and rand:
        means = bootstrap_means(rand, seed)
        out["candidate_vs_random"] = {
            "metric": "logit_drop",
            "diff_of_means": statistics.fmean(cand) - statistics.fmean(rand),
            "permutation_p": permutation_test(cand, rand, seed),
            "random_mean_ci95": [
                means[int(0.025 * len(means))],
                means[int(0.975 * len(means)) - 1],
            ],
        }
    return out


# --------------------------------------------------------------------- per prompt


def analyze_slug(cfg: SizeConfig, slug: str, label: dict, seed: int) -> dict:
    slug_dir = cfg.graph_dir / slug
    rhyme_step = label["rhyme_step"]

    contexts = step_contexts(slug_dir)
    step_nodes, _ = load_raw_step_nodes(slug_dir, min(INFLUENCE_GRID))
    if not step_nodes:
        raise ValueError(f"no graphs in {slug_dir}")

    measured = load_measurements(cfg, slug)
    if not measured.available:
        print(
            f"  [{cfg.size}/{slug}] no measurements in {cfg.results_path(slug).name} -- "
            "sweep reports candidate counts only"
        )

    # `n_layers` must match what tracing.py measured against: the sweep recomputes
    # candidates at every floor, so admitting last-layer features here would
    # invent cells that the GPU pass never measured and inflate `not_measured`.
    sweep = run_sweep(step_nodes, contexts, rhyme_step, measured, cfg.n_layers)

    # Populations are defined at the shipped influence threshold; the sweep above
    # is what varies it. Mixing the two would make near-miss membership depend on
    # a value that is itself under test.
    step_features, step_total = filter_at(step_nodes, INFLUENCE_THRESHOLD)
    timeline, percentiles = build_timeline(step_features, step_total)
    stats = feature_stats(timeline, percentiles, rhyme_step, cfg.n_layers)
    rhyme_step_features = {(r["layer"], r["feat"]) for r in step_features.get(rhyme_step, [])}
    pops = split_populations(stats, rhyme_step_features, rhyme_step)
    control_selected = sample_matched_control(pops["candidates"], pops["rest"], RANDOM_SEED)

    results: dict[str, Any] = {
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
            "seed": seed,
            "fixed_k": FIXED_K,
            "grids": {
                "influence_threshold": list(INFLUENCE_GRID),
                "min_rhyme_percentile": list(PERCENTILE_GRID),
                "min_sustain": list(SUSTAIN_GRID),
            },
            "margins": {"sustain": SUSTAIN_MARGIN, "percentile": PERCENTILE_MARGIN},
            "shipped": {
                "influence_threshold": INFLUENCE_THRESHOLD,
                "min_rhyme_percentile": CANDIDATE_MIN_RHYME_PERCENTILE,
                "min_sustain": CANDIDATE_MIN_SUSTAIN,
            },
        },
        "population_counts": {k: len(v) for k, v in pops.items()},
        "sweep": sweep,
    }

    print(
        f"\n{'=' * 70}\n[{cfg.size}/{slug}] {label['label']}  "
        f"{label['rhyme_word']} vs {label['target_word']} @ step {rhyme_step}\n{'=' * 70}"
    )
    print(
        f"  candidates={len(pops['candidates'])}  near_misses={len(pops['near_misses'])}  "
        f"other={len(pops['rest'])}  execution={len(pops['execution'])}"
    )
    for row in sorted(sweep, key=lambda r: (r["min_rhyme_percentile"], r["min_sustain"]))[
        :TOP_N_REPORTED
    ]:
        top = row["top_drops"]
        best = f"{top[0]['logit_drop']:+.3f}" if top else "n/a"
        print(
            f"    inf={row['influence_threshold']:<8} pct={row['min_rhyme_percentile']:<5} "
            f"sus={row['min_sustain']:<4} n={row['n_candidates']:<4} "
            f"measured={row['n_measured']} top_logit_drop={best}"
        )

    if not measured.available:
        for key in ("candidate_baseline", "near_miss", "random_control", "comparison"):
            results[key] = {"skipped": "no measurements available"}
        return results

    groups = {
        name: measured.population(name) for name in ("candidate", "near_miss", "random_control")
    }
    results["candidate_baseline"] = {
        "results": groups["candidate"],
        "summary": summarize(groups["candidate"], len(pops["candidates"])),
    }
    results["near_miss"] = {
        "results": groups["near_miss"],
        "summary": summarize(groups["near_miss"], len(pops["near_misses"])),
    }
    results["random_control"] = {
        "seed": seed,
        "results": groups["random_control"],
        # Recomputed, not counted from the measured rows: the draw is
        # deterministic given the seed, so this is the same set tracing.py
        # measured and makes the coverage check meaningful rather than tautological.
        "summary": summarize(groups["random_control"], len(control_selected)),
    }
    results["comparison"] = compare(groups, seed)
    # Both suppression points, reported side by side; neither is designated
    # primary here. `first_step` is defined against INFLUENCE_THRESHOLD and so
    # drifts as the sweep varies it -- the `measured_at_other_position` counts in
    # the sweep are the direct measurement of that drift.
    results["by_step_key"] = {
        key: (lambda sub: summarize(sub, len({(r["layer"], r["feat"]) for r in sub})))(
            [r for r in measured.rows if key in r.get("step_keys", [])]
        )
        for key in STEP_KEYS
    }

    cmp_block = results["comparison"].get("candidate_vs_random")
    if cmp_block:
        print(
            f"  candidate vs random: diff={cmp_block['diff_of_means']:+.4f} logit  "
            f"p={cmp_block['permutation_p']}"
        )
    return results


# --------------------------------------------------------------------------- entry


def run(cfg: SizeConfig, slugs: list[str] | None, seed: int) -> None:
    if not LABELS_PATH.exists():
        raise SystemExit(f"{LABELS_PATH} missing -- run `python experiment/rhyme_labels.py` first")
    if not cfg.graph_dir.exists():
        raise SystemExit(f"{cfg.graph_dir} missing -- generate graphs for {cfg.size} first")

    labels = {(r["size"], r["slug"]): r for r in json.loads(LABELS_PATH.read_text())}

    available = sorted(p.name for p in cfg.graph_dir.iterdir() if p.is_dir())
    targets = [s for s in (slugs or available) if s in available]
    for s in slugs or []:
        if s not in available:
            print(f"  [{cfg.size}] {s}: no graphs -- skipping")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for slug in targets:
        label = labels.get((cfg.size, slug))
        if label is None:
            print(f"  [{cfg.size}] {slug}: no rhyme label -- skipping")
            continue
        if label["rhyme_step"] is None:
            print(f"  [{cfg.size}] {slug}: {label['label']} -- skipping")
            continue

        results = analyze_slug(cfg, slug, label, seed)

        out = OUT_DIR / f"threshold_sensitivity_{cfg.size}_{slug}.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"  wrote {out.relative_to(Path(__file__).parent)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--slugs", nargs="*", help="prompt slugs (default: all with graphs)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    run(CONFIGS[args.size], slugs=args.slugs, seed=args.seed)


if __name__ == "__main__":
    main()
