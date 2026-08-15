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

EXPERIMENT = Path(__file__).parent
REPO = EXPERIMENT.parent
PROMPT_SET_PATH = REPO / "tools" / "prompt_set.json"
LABELS_PATH = EXPERIMENT / "rhyme_labels.json"
# Must match tracing.py's RESULTS_DIR.
RESULTS_DIR = EXPERIMENT / "tracing"

SIZES = ["270m", "1b", "4b"]
TOP_N = 15
RULE = "=" * 78


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
# scipy is not a declared dependency (pyproject.toml:7-26) and a slope plus a
# rank correlation does not justify adding one, so both are computed here.


def spearman(x: list[float], y: list[float]) -> float:
    """Rank correlation, averaging ranks over ties."""
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    # A metric that is constant across prompts has zero rank variance; corrcoef
    # would divide by zero and warn rather than just returning nan.
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = arr.argsort()
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(len(arr), dtype=float)
    # Average ranks within tied groups so ties do not create spurious ordering.
    for value in np.unique(arr):
        tied = arr == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def ols(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """(slope, intercept, r_squared) for a least-squares line."""
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xa.std() == 0 or ya.std() == 0:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(xa, ya, 1)
    r = np.corrcoef(xa, ya)[0, 1]
    return float(slope), float(intercept), float(r**2)


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


def section_suppression(results, sizes, labels) -> None:
    """Causal suppression, tolerating every shape run_interventions can emit."""
    print(f"\n{RULE}\nSUPPRESSION EFFECTS\n{RULE}")
    for size in sizes:
        rows = select(results, size, labels)
        if not rows:
            print(f"\n{size}: no results")
            continue
        print(f"\n{size}:")
        pooled_max: list[float] = []
        for slug, p in rows:
            iv = p.get("interventions")
            if iv is None:
                print(f"  {slug:10s} {label_of(p):12s} not run (--no-interventions)")
                continue
            if "skipped" in iv:
                print(f"  {slug:10s} {label_of(p):12s} skipped: {iv['skipped']}")
                continue
            for key in ("peak_step", "first_step"):
                block = iv.get(key)
                if not block or not block.get("results"):
                    continue
                drops = [r["prob_drop"] for r in block["results"]]
                agg = block.get("summary", {}).get("aggregate", {})
                best = max(block["results"], key=lambda r: r["prob_drop"])
                if key == "peak_step":
                    pooled_max.append(max(drops))
                print(
                    f"  {slug:10s} {label_of(p):12s} {key:10s} "
                    f"n={block['n_measured']:3d} max={max(drops):.4f} "
                    f"avg={agg.get('avg_prob_drop', mean(drops)):.4f} "
                    f"pos={sum(1 for d in drops if d > 0):3d}  "
                    f"top=L{best['layer']}F{best['feat']}"
                )
        if pooled_max:
            print(f"  {'POOLED':10s} {'':12s} max prob_drop across prompts: {describe(pooled_max)}")


def section_rhyme_availability(results, sizes, rhymes_all, labels) -> None:
    """The grid's design variable: does planning scale with rhyme availability?

    Descriptive only. With n<=10 usable points per size these coefficients are
    not powered for significance testing, and no p-values are reported.
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

        for name, fn in metrics.items():
            ys = [fn(p) for _, p in fit_rows]
            slope, _, r2 = ols(xs, ys)
            rho = spearman(xs, ys)
            print(
                f"  {name:14s} slope={slope:8.3f} per decade  "
                f"r2={r2:5.3f}  spearman={rho:6.3f}  (n={len(ys)})"
            )

        print(f"  {'per prompt:':14s}")
        for slug, p in sorted(fit_rows, key=lambda kv: rhymes_all[kv[0]]):
            vals = "  ".join(f"{n}={fn(p):7.2f}" for n, fn in metrics.items())
            print(f"    {slug:10s} rhymes_all={rhymes_all[slug]:4d} {label_of(p):12s} {vals}")
        for slug, p in anchor:
            vals = "  ".join(f"{n}={fn(p):7.2f}" for n, fn in metrics.items())
            print(f"    {slug:10s} {'anchor':>15s} {label_of(p):12s} {vals}")
        print()
    print("Descriptive only -- n<=10 per size is not powered for significance testing.")


def section_summary(results, sizes, labels) -> None:
    print(f"\n{RULE}\nSUMMARY\n{RULE}")
    header = (
        f"{'Model':<8}{'Prompts':>8}{'Rhymes':>8}{'Unique':>9}"
        f"{'Cands':>8}{'Spikes':>8}{'Plan %':>9}{'MaxDrop':>9}"
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

        drops = []
        for _, p in rows:
            block = (p.get("interventions") or {}).get("peak_step")
            if block and block.get("results"):
                drops.append(max(r["prob_drop"] for r in block["results"]))
        drop = f"{max(drops):.4f}" if drops else "--"

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
    section_suppression(results, sizes, args.labels)
    section_rhyme_availability(results, sizes, rhymes_all, args.labels)
    section_summary(results, sizes, args.labels)


if __name__ == "__main__":
    main()
