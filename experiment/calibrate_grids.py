"""Derives INFLUENCE_GRID/SUSTAIN_GRID and their shipped defaults from the
current graph corpus and writes them to grid_calibration.json, which
tracing.py reads at import time.

    python experiment/calibrate_grids.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tracing import CONFIGS, LABELS_PATH, build_timeline, feature_stats, filter_at, load_raw_step_nodes

CALIBRATION_PATH = Path(__file__).parent / "grid_calibration.json"
PERCENTILES = (10, 30, 50, 70, 90)


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    frac = k - lo
    return values[lo] + (values[hi] - values[lo]) * frac


def main() -> None:
    labels = json.loads(LABELS_PATH.read_text())
    label_map = {(r["size"], r["slug"]): r for r in labels}

    # Pass 1: influence distribution, read once and cached -- also needed to
    # know p50 (the shipped INFLUENCE_THRESHOLD) before pass 2 can filter at it.
    influence_vals: list[float] = []
    corpus_keys: list[str] = []
    cached: list[tuple[str, str, dict, dict | None]] = []

    for size, cfg in CONFIGS.items():
        for slug_dir in sorted(cfg.graph_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            slug = slug_dir.name
            corpus_keys.append(f"{size}/{slug}")

            step_nodes, _ = load_raw_step_nodes(slug_dir, 0.0)
            for rows in step_nodes.values():
                influence_vals.extend(r["influence"] for r in rows)
            cached.append((size, slug, step_nodes, label_map.get((size, slug))))

    influence_percentiles = {f"p{p}": percentile(influence_vals, p) for p in PERCENTILES}

    # Pass 2: sustain_ratio, filtered at the influence p50 -- the same floor the
    # real pipeline applies (INFLUENCE_THRESHOLD) before computing sustain_ratio.
    # Calibrating against the unfiltered (floor 0.0) population would quantify a
    # different distribution than the one the shipped cutoff is actually applied to.
    sustain_vals: list[float] = []
    influence_floor = influence_percentiles["p50"]
    for size, slug, step_nodes, rec in cached:
        if rec is None or rec.get("rhyme_step") is None:
            continue
        rhyme_step = rec["rhyme_step"]
        cfg = CONFIGS[size]
        step_features, step_total = filter_at(step_nodes, influence_floor)
        timeline, percentiles = build_timeline(step_features, step_total)
        stats = feature_stats(timeline, percentiles, rhyme_step, cfg.n_layers)
        rhyme_step_feats = {(r["layer"], r["feat"]) for r in step_features.get(rhyme_step, [])}
        for s in stats:
            if s["peak_step"] < rhyme_step and s["feat_key"] in rhyme_step_feats:
                sustain_vals.append(s["sustain_ratio"])

    marker = hashlib.sha256(
        f"{sorted(corpus_keys)}|{len(influence_vals)}|{len(sustain_vals)}".encode()
    ).hexdigest()[:16]

    calibration = {
        "corpus_marker": marker,
        "n_prompts": len(corpus_keys),
        "n_influence_samples": len(influence_vals),
        "n_sustain_samples": len(sustain_vals),
        "influence": influence_percentiles,
        "sustain": {f"p{p}": percentile(sustain_vals, p) for p in PERCENTILES},
    }
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2) + "\n")
    print(f"wrote {CALIBRATION_PATH} (marker {marker}, {len(corpus_keys)} prompts)")
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
