"""Cross-model comparison over the 11-prompt grid.

Reads the per-prompt outputs of the tracing stage
(`circuit_tracing_results_{size}_{slug}.json`) and reports how planning-feature
structure varies with model size and with rhyme availability.

Two things this deliberately does NOT do:

* It never intersects feature indices across model sizes. 270M/1B/4B use
  separately-trained transcoders (`gemma-scope-2-{size}-it`), so feature k in
  one has no relation to feature k in another, and the differing layer counts
  (18/26/34) would bias any such intersection toward low layers. Feature
  persistence is reported WITHIN a size, across prompts, where the transcoder
  is fixed and the indices mean the same thing.

* It does not filter to rhyming prompts. Every statistic is split by rhyme
  label, so 270M's 0/11 stays visible as a measured negative rather than an
  empty table.

    python experiment/comparing.py                    # all sizes on disk
    python experiment/comparing.py --sizes 270m 1b    # selected sizes
    python experiment/comparing.py --labels rhyme     # headline subset only
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

EXPERIMENT = Path(__file__).parent
REPO = EXPERIMENT.parent
PROMPT_SET_PATH = REPO / "tools" / "prompt_set.json"
LABELS_PATH = EXPERIMENT / "rhyme_labels.json"
# Must match tracing.py's RESULTS_DIR.
RESULTS_DIR = EXPERIMENT / "tracing"

SIZES = ["270m", "1b", "4b"]
TOP_N = 15
RULE = "=" * 78
# Headline suppression statistic. Both a mean over every candidate and a max
# scale with the candidate count, which scales with continuation length, so
# neither is comparable across prompts. Fixed-k fixes the draw; prompts with
# fewer than k measured features are excluded and reported as such.
FIXED_K = 10
JSON_OUT = EXPERIMENT / "comparison.json"


# ------------------------------------------------------------------- loading


def load_all(sizes: list[str]) -> tuple[dict[tuple[str, str], dict], dict[str, int | None]]:
    """Every per-prompt result, keyed (size, slug), joined with rhymes_all.

    Returns (results, rhymes_all_by_slug). Prompts present in prompt_set.json
    but missing a result file are reported -- a size with 10 of 11 prompts is a
    different claim from one with 11.
    """
    prompt_set = {r["slug"]: r for r in json.loads(PROMPT_SET_PATH.read_text())}
    rhymes_all = {slug: r.get("rhymes_all") for slug, r in prompt_set.items()}

    results: dict[tuple[str, str], dict] = {}
    for size in sizes:
        found = set()
        for path in sorted(RESULTS_DIR.glob(f"circuit_tracing_results_{size}_*.json")):
            slug = path.name[len(f"circuit_tracing_results_{size}_") : -len(".json")]
            if slug not in prompt_set:
                print(f"  [{size}] {slug}: not in prompt_set.json -- skipping")
                continue
            results[(size, slug)] = json.loads(path.read_text())
            found.add(slug)
        missing = sorted(set(prompt_set) - found)
        status = f"{len(found)}/{len(prompt_set)} prompts"
        print(f"[{size}] {status}" + (f"  MISSING: {', '.join(missing)}" if missing else ""))

    return results, rhymes_all


def label_of(payload: dict) -> str:
    """Rhyme label, mirrored into every result by tracing.py:298."""
    return payload["config"].get("rhyme_label", "unknown")


def select(
    results: dict[tuple[str, str], dict], size: str, labels: list[str] | None
) -> list[tuple[str, dict]]:
    """[(slug, payload)] for one size, optionally restricted to given labels."""
    rows = [(slug, p) for (sz, slug), p in results.items() if sz == size]
    if labels:
        rows = [(s, p) for s, p in rows if label_of(p) in labels]
    return sorted(rows)


# ---------------------------------------------------------------- statistics
#
# scipy.stats rather than hand-rolled: these numbers go in a paper, and a
# reviewer should not have to audit our correlation code. Both wrappers exist
# only to return nan quietly on degenerate input (n < 3, or a metric that is
# constant across prompts) instead of raising or warning.

NAN = float("nan")


def spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    """(rho, p) via scipy.stats.spearmanr; ties get averaged ranks."""
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return NAN, NAN
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def ols(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """(slope, r_squared, p) via scipy.stats.linregress."""
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return NAN, NAN, NAN
    fit = stats.linregress(x, y)
    return float(fit.slope), float(fit.rvalue**2), float(fit.pvalue)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def describe(values: list[float]) -> str:
    if not values:
        return "n=0"
    return f"n={len(values):2d}  mean={mean(values):8.3f}  min={min(values):8.3f}  max={max(values):8.3f}"


# ------------------------------------------------------------------ sections


def section_coverage(results, sizes, labels) -> None:
    print(f"\n{RULE}\nCOVERAGE AND RHYME LABELS\n{RULE}")
    for size in sizes:
        rows = select(results, size, None)
        if not rows:
            print(f"\n{size}: no results")
            continue
        counts = Counter(label_of(p) for _, p in rows)
        n_rhyme = counts.get("rhyme", 0)
        print(f"\n{size}: {n_rhyme}/{len(rows)} rhyme   {dict(counts)}")
        for slug, p in rows:
            cfg = p["config"]
            print(
                f"  {slug:10s} {label_of(p):12s} "
                f"{str(cfg['rhyme_word']):12s} vs {str(cfg['target_word']):10s} "
                f"step={cfg['rhyme_step']}"
            )
    if labels:
        print(f"\n(subsequent sections restricted to labels: {', '.join(labels)})")


def section_timing(results, sizes, labels) -> None:
    """Peak-step timing, expressed as lead before the rhyme step.

    Raw peak_step is not comparable across prompts -- rhyme_step differs -- so
    the reported quantity is (rhyme_step - peak_step), i.e. how many steps
    before the rhyme a feature peaked.
    """
    print(f"\n{RULE}\nPEAK TIMING (steps before the rhyme step)\n{RULE}")
    for size in sizes:
        rows = select(results, size, labels)
        if not rows:
            print(f"\n{size}: no results")
            continue
        print(f"\n{size}:")
        all_leads: list[float] = []
        for slug, p in rows:
            rhyme_step = p["config"]["rhyme_step"]
            leads = [rhyme_step - c["peak_step"] for c in p["candidates"]]
            all_leads += leads
            dist = Counter(leads)
            top = " ".join(f"+{k}:{v}" for k, v in sorted(dist.items(), reverse=True)[:6])
            print(f"  {slug:10s} {label_of(p):12s} {describe(leads)}   {top}")
        print(f"  {'POOLED':10s} {'':12s} {describe(all_leads)}")


def section_sustain(results, sizes, labels) -> None:
    print(f"\n{RULE}\nSUSTAIN RATIO\n{RULE}")
    for size in sizes:
        rows = select(results, size, labels)
        pooled = [c["sustain_ratio"] for _, p in rows for c in p["candidates"]]
        print(f"\n{size}:  {describe(pooled)}")
        for slug, p in rows:
            vals = [c["sustain_ratio"] for c in p["candidates"]]
            print(f"  {slug:10s} {label_of(p):12s} {describe(vals)}")


def section_planning_execution(results, sizes, labels) -> None:
    print(f"\n{RULE}\nPLANNING VS EXECUTION FEATURES\n{RULE}")
    for size in sizes:
        rows = select(results, size, labels)
        if not rows:
            print(f"\n{size}: no results")
            continue
        print(f"\n{size}:")
        pcts = []
        for slug, p in rows:
            st = p["statistics"]
            plan, execu = st["n_planning_features"], st["n_execution_features"]
            total = plan + execu
            pct = 100 * plan / total if total else 0.0
            pcts.append(pct)
            print(
                f"  {slug:10s} {label_of(p):12s} planning={plan:6d} "
                f"execution={execu:6d}  planning={pct:5.1f}%"
            )
        print(f"  {'MEAN':10s} {'':12s} planning={mean(pcts):5.1f}% across {len(pcts)} prompts")


def section_bands(results, sizes, labels) -> None:
    """Temporal clustering, using the cutoffs tracing.py already stored."""
    print(f"\n{RULE}\nTEMPORAL CLUSTERING (EARLY / MID / LATE)\n{RULE}")
    for size in sizes:
        rows = select(results, size, labels)
        if not rows:
            print(f"\n{size}: no results")
            continue
        print(f"\n{size}:")
        totals: Counter = Counter()
        for slug, p in rows:
            counts = p["statistics"].get("band_counts", {})
            totals.update(counts)
            n = sum(counts.values())
            parts = " ".join(
                f"{k}={v}({100 * v / n:4.1f}%)" if n else f"{k}={v}" for k, v in counts.items()
            )
            print(
                f"  {slug:10s} {label_of(p):12s} "
                f"[cut {p['config']['early_cutoff']}/{p['config']['mid_cutoff']}]  {parts}"
            )
        grand = sum(totals.values())
        if grand:
            parts = " ".join(f"{k}={v}({100 * v / grand:4.1f}%)" for k, v in totals.items())
            print(f"  {'POOLED':10s} {'':12s} {parts}")


def section_persistence(results, sizes, labels) -> None:
    """Feature reuse ACROSS PROMPTS, WITHIN a size.

    Valid because one size uses one transcoder set, so (layer, feat) means the
    same thing in every prompt. The cross-size version of this comparison is
    not computable and is deliberately absent.
    """
    print(f"\n{RULE}\nWITHIN-SIZE FEATURE PERSISTENCE (across prompts)\n{RULE}")
    print("Cross-size feature identity is not computable: 270M/1B/4B use")
    print("separately-trained transcoders, so feature indices are unrelated.\n")
    for size in sizes:
        rows = select(results, size, labels)
        if not rows:
            print(f"{size}: no results\n")
            continue
        seen: dict[tuple[int, int], set[str]] = defaultdict(set)
        for slug, p in rows:
            for c in p["candidates"]:
                seen[(c["layer"], c["feat"])].add(slug)
        n_prompts = len(rows)
        recur = {k: v for k, v in seen.items() if len(v) >= 2}
        half = {k: v for k, v in seen.items() if len(v) >= max(2, math.ceil(n_prompts / 2))}
        print(f"{size}:  {n_prompts} prompts, {len(seen)} distinct candidate features")
        print(
            f"  in >=2 prompts:            {len(recur):5d} ({100 * len(recur) / max(len(seen), 1):.1f}%)"
        )
        print(f"  in >=half ({math.ceil(n_prompts / 2)} prompts):      {len(half):5d}")
        for (layer, feat), slugs in sorted(recur.items(), key=lambda kv: -len(kv[1]))[:TOP_N]:
            print(f"    L{layer:2d} F{feat:6d}  {len(slugs):2d} prompts  {' '.join(sorted(slugs))}")
        print()


def suppression_rows(payload: dict) -> list[dict] | None:
    """Measured rows from one result file, or None if there are none.

    Tolerates every shape `run_interventions` can emit: absent (analysis-only
    run), `{"skipped": ...}`, or a full block.
    """
    iv = payload.get("interventions")
    if not isinstance(iv, dict) or "results" not in iv:
        return None
    return iv["results"]


def best_per_feature(rows: list[dict]) -> list[dict]:
    """One row per (layer, feat): its strongest suppression across positions.

    A feature is measured at up to two positions -- its `peak_step` and its
    `first_step` -- so a top-k over raw rows could spend two of its k slots on
    one feature, and a row count is not a feature count.
    """
    best: dict[tuple[int, int], dict] = {}
    for r in rows:
        key = (r["layer"], r["feat"])
        if key not in best or r["logit_drop"] > best[key]["logit_drop"]:
            best[key] = r
    return list(best.values())


def fixed_k(rows: list[dict], k: int = FIXED_K) -> float | None:
    """Mean `logit_drop` over the top k features, or None if fewer than k.

    `logit_drop` rather than `prob_drop`: probability cannot register an effect
    once the target is already near zero, and `prob_drop` is bounded above by
    `original_prob` (`methodology_evidence.md` §1).
    """
    top = sorted(best_per_feature(rows), key=lambda r: -r["logit_drop"])[:k]
    return mean([r["logit_drop"] for r in top]) if len(top) == k else None


def section_suppression(results, sizes, labels) -> dict:
    print(f"\n{RULE}\nSUPPRESSION EFFECTS\n{RULE}")
    print(f"Headline statistic is mean logit_drop over the top {FIXED_K} measured")
    print("features. Max and n are descriptive: both grow with candidate count.\n")
    out: dict = {}
    for size in sizes:
        rows = select(results, size, labels)
        if not rows:
            print(f"\n{size}: no results")
            continue
        print(f"\n{size}:")
        per_size: dict = {}
        pooled_k: list[float] = []
        for slug, p in rows:
            measured = suppression_rows(p)
            if measured is None:
                iv = p.get("interventions")
                why = iv["skipped"] if isinstance(iv, dict) and "skipped" in iv else "not run"
                print(f"  {slug:10s} {label_of(p):12s} {why}")
                per_size[slug] = {"skipped": why}
                continue

            entry: dict = {}
            for pop in ("candidate", "superset", "near_miss", "random_control"):
                sub = [r for r in measured if pop in r.get("populations", [])]
                if not sub:
                    continue
                k = fixed_k(sub)
                per_feat = best_per_feature(sub)
                entry[pop] = {
                    "n_features": len(per_feat),
                    "n_rows": len(sub),
                    f"mean_logit_drop_top{FIXED_K}": k,
                    "max_logit_drop": max(r["logit_drop"] for r in per_feat),
                    "mean_prob_drop": mean([r["prob_drop"] for r in per_feat]),
                    "n_positive": sum(1 for r in per_feat if r["logit_drop"] > 0),
                }
            for key in ("peak_step", "first_step"):
                sub = [r for r in measured if key in r.get("step_keys", [])]
                if sub:
                    entry[key] = {
                        "n_features": len(best_per_feature(sub)),
                        f"mean_logit_drop_top{FIXED_K}": fixed_k(sub),
                    }

            cand = entry.get("candidate", {})
            ctrl = entry.get("random_control", {})
            ck = cand.get(f"mean_logit_drop_top{FIXED_K}")
            rk = ctrl.get(f"mean_logit_drop_top{FIXED_K}")
            if ck is not None:
                pooled_k.append(ck)
            print(
                f"  {slug:10s} {label_of(p):12s} "
                f"cand n={cand.get('n_features', 0):4d} "
                f"top{FIXED_K}={'  n/a' if ck is None else f'{ck:+6.3f}'}   "
                f"ctrl n={ctrl.get('n_features', 0):4d} "
                f"top{FIXED_K}={'  n/a' if rk is None else f'{rk:+6.3f}'}"
            )
            per_size[slug] = entry

        if pooled_k:
            print(
                f"  {'POOLED':10s} {'':12s} candidate top{FIXED_K} logit drop: {describe(pooled_k)}"
            )
        out[size] = per_size
    return out


def section_rhyme_availability(results, sizes, rhymes_all, labels) -> dict:
    """The grid's design variable: does planning scale with rhyme availability?

    Descriptive only. p-values come from scipy and are printed for
    completeness, but n<=10 per size is underpowered and the three metrics x
    three sizes are uncorrected for multiple comparisons.
    """
    print(f"\n{RULE}\nPLANNING VS RHYME AVAILABILITY (per size)\n{RULE}")
    print("Regressed on log10(rhymes_all). The `original` anchor has no")
    print("rhymes_all and is excluded from the fit, reported separately.\n")

    metrics = {
        "planning %": lambda p: (
            100
            * p["statistics"]["n_planning_features"]
            / max(
                p["statistics"]["n_planning_features"] + p["statistics"]["n_execution_features"], 1
            )
        ),
        "n_candidates": lambda p: float(p["statistics"]["n_candidates"]),
        "mean lead": lambda p: mean(
            [p["config"]["rhyme_step"] - c["peak_step"] for c in p["candidates"]]
        ),
    }

    fits: dict = {}
    for size in sizes:
        rows = select(results, size, labels)
        if not rows:
            print(f"{size}: no results\n")
            continue

        fit_rows = [(s, p) for s, p in rows if rhymes_all.get(s)]
        anchor = [(s, p) for s, p in rows if not rhymes_all.get(s)]
        xs = [math.log10(rhymes_all[s]) for s, _ in fit_rows]

        print(f"{size}:  n={len(fit_rows)} scoreable prompts")
        if len(fit_rows) < 3:
            print("  too few points to fit\n")
            continue

        fits[size] = {}
        for name, fn in metrics.items():
            ys = [fn(p) for _, p in fit_rows]
            slope, r2, p_lin = ols(xs, ys)
            rho, p_rho = spearman(xs, ys)
            fits[size][name] = {
                "n": len(ys),
                "slope_per_decade": slope,
                "r_squared": r2,
                "p_linregress": p_lin,
                "spearman_rho": rho,
                "p_spearman": p_rho,
                "per_prompt": {s: fn(p) for s, p in fit_rows},
            }
            print(
                f"  {name:14s} slope={slope:8.3f} per decade  "
                f"r2={r2:5.3f} p={p_lin:5.3f}  "
                f"spearman={rho:6.3f} p={p_rho:5.3f}  (n={len(ys)})"
            )

        print(f"  {'per prompt:':14s}")
        for slug, p in sorted(fit_rows, key=lambda kv: rhymes_all[kv[0]]):
            vals = "  ".join(f"{n}={fn(p):7.2f}" for n, fn in metrics.items())
            print(f"    {slug:10s} rhymes_all={rhymes_all[slug]:4d} {label_of(p):12s} {vals}")
        for slug, p in anchor:
            vals = "  ".join(f"{n}={fn(p):7.2f}" for n, fn in metrics.items())
            print(f"    {slug:10s} {'anchor':>15s} {label_of(p):12s} {vals}")
        print()
    print(
        "Descriptive only. p-values are shown for completeness but n<=10 per size\n"
        "is underpowered, and three metrics x three sizes are uncorrected for\n"
        "multiple comparisons -- do not read them as significance tests."
    )
    return fits


def section_summary(results, sizes, labels) -> None:
    print(f"\n{RULE}\nSUMMARY\n{RULE}")
    header = (
        f"{'Model':<8}{'Prompts':>8}{'Rhymes':>8}{'Unique':>9}"
        f"{'Cands':>8}{'Spikes':>8}{'Plan %':>9}{'Top10Lgt':>9}"
    )
    print(header)
    print("-" * len(header))
    for size in sizes:
        rows = select(results, size, labels)
        if not rows:
            continue
        n_rhyme = sum(1 for _, p in rows if label_of(p) == "rhyme")
        uniq = sum(p["statistics"]["n_unique_features"] for _, p in rows)
        cands = sum(p["statistics"]["n_candidates"] for _, p in rows)
        spikes = sum(p["statistics"]["n_early_spikes"] for _, p in rows)
        plan = sum(p["statistics"]["n_planning_features"] for _, p in rows)
        execu = sum(p["statistics"]["n_execution_features"] for _, p in rows)
        plan_pct = 100 * plan / (plan + execu) if plan + execu else 0.0

        # Per-prompt top-k means, then described across prompts. Reporting
        # max(max) instead let a single prompt stand in for a whole size.
        ks = []
        for _, p in rows:
            measured = suppression_rows(p)
            if measured:
                k = fixed_k([r for r in measured if "candidate" in r.get("populations", [])])
                if k is not None:
                    ks.append(k)
        drop = f"{mean(ks):+.4f}" if ks else "--"

        print(
            f"{size:<8}{len(rows):>8}{n_rhyme:>8}{uniq:>9}"
            f"{cands:>8}{spikes:>8}{plan_pct:>9.1f}{drop:>9}"
        )


# -------------------------------------------------------------------- entry


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-model comparison over the prompt grid")
    ap.add_argument("--sizes", nargs="*", default=SIZES, help="model sizes (default: all)")
    ap.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="restrict analysis to these rhyme labels, e.g. --labels rhyme near_rhyme",
    )
    args = ap.parse_args()

    sizes = [s for s in args.sizes if s in SIZES]
    results, rhymes_all = load_all(sizes)
    if not results:
        raise SystemExit(
            f"no circuit_tracing_results_*.json in {RESULTS_DIR} -- "
            "run experiment/tracing-*.py first"
        )

    section_coverage(results, sizes, args.labels)
    section_timing(results, sizes, args.labels)
    section_sustain(results, sizes, args.labels)
    section_planning_execution(results, sizes, args.labels)
    section_bands(results, sizes, args.labels)
    section_persistence(results, sizes, args.labels)
    suppression = section_suppression(results, sizes, args.labels)
    fits = section_rhyme_availability(results, sizes, rhymes_all, args.labels)
    section_summary(results, sizes, args.labels)

    # Everything above also goes to disk. Stdout alone left no artifact to diff
    # between runs or to cite a number from without re-running the whole stage.
    dump = {
        "sizes": sizes,
        "labels_filter": args.labels,
        "fixed_k": FIXED_K,
        "rhymes_all": rhymes_all,
        "provenance": {
            f"{sz}/{slug}": p.get("provenance") for (sz, slug), p in sorted(results.items())
        },
        "per_prompt": {
            f"{sz}/{slug}": {
                "rhyme_label": label_of(p),
                "config": p["config"],
                "statistics": p["statistics"],
                "population_counts": p.get("population_counts"),
                "position_diagnostics": p.get("position_diagnostics"),
            }
            for (sz, slug), p in sorted(results.items())
        },
        "suppression": suppression,
        "rhyme_availability_fits": fits,
    }
    JSON_OUT.write_text(json.dumps(dump, indent=2, default=str))
    print(f"\nwrote {JSON_OUT}")


if __name__ == "__main__":
    main()
