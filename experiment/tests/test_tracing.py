"""Golden fixtures for the GPU-free half of the tracing pipeline.

`experiment/` has no CI coverage, and the bug these exist to catch was a silent
one: suppression interventions were being applied at a *generation step index*
where the library expects a *token position*, so every measurement in the
pipeline ablated a token inside the prompt. Nothing crashed. The numbers just
described the wrong thing.

The fixtures below build small synthetic graphs with known `prompt_tokens` and
known `ctx_idx`, so the derived positions have an answer to be checked against.
"""

from __future__ import annotations

import json

import pytest

import tracing
from tracing import (
    CANDIDATE_MIN_RHYME_PERCENTILE,
    build_measurement_set,
    build_timeline,
    collect_ctx_idx,
    feature_stats,
    filter_at,
    load_raw_step_nodes,
    parse_node_ids,
    sample_matched_control,
    select_candidates,
    split_populations,
    step_contexts,
)

BASE_NTOK = 20  # tokens in the prompt before any generation
N_STEPS = 6
RHYME_STEP = 4
# The fixture's features live at layers 1-4, so a last layer of 5 leaves every
# existing expectation untouched: the layer filter is exercised by the two
# dedicated tests below instead of by silently reshaping the golden populations.
N_LAYERS = 6


def make_node(layer: int, feat: int, influence: float, ctx_idx: int) -> dict:
    return {
        "feature_type": "cross layer transcoder",
        "jsNodeId": f"{layer}_{feat}-0",
        # Deliberately present and deliberately wrong-as-a-feature-index: this is
        # a cantor pairing in the real export, and parse_node_ids must not read it.
        "feature": 999999,
        "layer": layer,
        "influence": influence,
        "ctx_idx": ctx_idx,
    }


@pytest.fixture
def slug_dir(tmp_path):
    """Six steps. `ntok` is linear in the step; nodes sit at the final position.

    Feature (1, 100) peaks early and holds -> a planning candidate.
    Feature (2, 200) peaks at the rhyme step -> execution, never a candidate.
    Feature (3, 300) is weak throughout -> control pool.
    Feature (4, 400) sits just under both cutoffs -> near miss.
    """
    d = tmp_path / "fixture"
    d.mkdir()
    profiles = {
        (1, 100): [0.05, 0.30, 0.20, 0.18, 0.16, 0.10],
        (2, 200): [0.01, 0.02, 0.03, 0.05, 0.40, 0.30],
        (3, 300): [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        (4, 400): [0.02, 0.20, 0.10, 0.08, 0.05, 0.03],
    }
    for step in range(N_STEPS):
        ntok = BASE_NTOK + step
        nodes = [
            make_node(layer, feat, vals[step], ntok - 1)
            for (layer, feat), vals in profiles.items()
            if vals[step] > 0
        ]
        payload = {
            "metadata": {
                "prompt": "<bos><start_of_turn>user\n" + "w" * step,
                "prompt_tokens": list(range(ntok)),
            },
            "nodes": nodes,
        }
        (d / f"step-{step:02d}-tok.json").write_text(json.dumps(payload))
    return d


# ----------------------------------------------------------------- node parsing


def test_parse_node_ids_reads_jsnodeid():
    assert parse_node_ids(make_node(7, 42, 0.1, 3)) == (7, 42)


def test_parse_node_ids_refuses_the_cantor_fallback():
    """`feature` is cantor_pairing(layer, feat), not a feature index.

    The old fallback returned it anyway, which produced wrong-but-plausible
    identities for every node in a graph whose export format had changed.
    """
    bad = {"feature_type": "transcoder", "jsNodeId": "", "layer": 3, "feature": 12345}
    with pytest.raises(ValueError, match="cantor"):
        parse_node_ids(bad)


# --------------------------------------------------------------------- contexts


def test_step_contexts_positions_are_linear_in_step(slug_dir):
    ctx = step_contexts(slug_dir)
    assert len(ctx) == N_STEPS
    for step in range(N_STEPS):
        assert ctx[step]["ntok"] == BASE_NTOK + step
        assert ctx[step]["position"] == BASE_NTOK + step - 1


def test_step_contexts_rejects_graphs_without_metadata(tmp_path):
    d = tmp_path / "broken"
    d.mkdir()
    (d / "step-00-x.json").write_text(json.dumps({"nodes": []}))
    with pytest.raises(ValueError, match="prompt_tokens"):
        step_contexts(d)


# ------------------------------------------------------------------ populations


def build(slug_dir):
    contexts = step_contexts(slug_dir)
    step_nodes, _ = load_raw_step_nodes(slug_dir, min(tracing.INFLUENCE_GRID))
    step_features, step_total = filter_at(step_nodes, tracing.INFLUENCE_THRESHOLD)
    timeline, percentiles = build_timeline(step_features, step_total)
    stats = feature_stats(timeline, percentiles, RHYME_STEP, N_LAYERS)
    rhyme_feats = {(r["layer"], r["feat"]) for r in step_features.get(RHYME_STEP, [])}
    return contexts, step_features, stats, rhyme_feats


def test_ctx_idx_is_carried_through(slug_dir):
    _, step_features, _, _ = build(slug_dir)
    got = collect_ctx_idx(step_features)
    assert got[(1, 100)][3] == BASE_NTOK + 3 - 1


def test_execution_features_are_excluded_from_the_control_pool(slug_dir):
    """The hazard that the position fix created.

    (2, 200) peaks at the rhyme step. Left in `rest`, the matched-random control
    could draw it, and its position would fall past the end of the measurement
    sequence -- an IndexError swallowed by the per-feature except, showing up
    only as an unexplained shortfall in n_measured.
    """
    _, _, stats, rhyme_feats = build(slug_dir)
    pops = split_populations(stats, rhyme_feats, RHYME_STEP)
    assert (2, 200) in {s["feat_key"] for s in pops["execution"]}
    for group in ("candidates", "near_misses", "rest"):
        assert all(s["peak_step"] < RHYME_STEP for s in pops[group])


def test_candidate_and_near_miss_land_where_expected(slug_dir):
    _, _, stats, rhyme_feats = build(slug_dir)
    pops = split_populations(stats, rhyme_feats, RHYME_STEP)
    assert (1, 100) in {s["feat_key"] for s in pops["candidates"]}
    assert (1, 100) not in {s["feat_key"] for s in pops["near_misses"]}


# --------------------------------------------------------------- measurement set


def test_positions_agree_with_ctx_idx_and_stay_in_bounds(slug_dir):
    contexts, step_features, stats, rhyme_feats = build(slug_dir)
    pops = split_populations(stats, rhyme_feats, RHYME_STEP)
    ctx_idx = collect_ctx_idx(step_features)
    rows, diag = build_measurement_set(
        {"candidate": pops["candidates"], "near_miss": pops["near_misses"]},
        contexts,
        ctx_idx,
        RHYME_STEP,
        N_LAYERS,
    )
    assert rows, "fixture should produce at least one measurable row"
    assert diag["ctx_idx_agreement"] == 1.0
    assert diag["measurement_length"] == BASE_NTOK + RHYME_STEP
    for r in rows:
        assert r["position"] < diag["measurement_length"]
        # The bug in one line: a position is not a step.
        assert r["position"] != r["step"]


def test_step_keys_are_deduplicated_by_position(slug_dir):
    """A feature whose first and peak step coincide is measured once, not twice.

    Roughly half of all real candidates are in this case (50.8%, n=2560), and the
    old loop ran the identical intervention twice for every one of them.
    """
    contexts, step_features, stats, rhyme_feats = build(slug_dir)
    entry = next(s for s in stats if s["feat_key"] == (1, 100))
    entry = {**entry, "first_step": entry["peak_step"]}
    rows, _ = build_measurement_set(
        {"candidate": [entry]}, contexts, collect_ctx_idx(step_features), RHYME_STEP, N_LAYERS
    )
    assert len(rows) == 1
    assert sorted(rows[0]["step_keys"]) == ["first_step", "peak_step"]


def test_population_tags_merge_rather_than_overwrite(slug_dir):
    contexts, step_features, stats, rhyme_feats = build(slug_dir)
    entry = next(s for s in stats if s["feat_key"] == (1, 100))
    rows, _ = build_measurement_set(
        {"candidate": [entry], "superset": [entry]},
        contexts,
        collect_ctx_idx(step_features),
        RHYME_STEP,
        N_LAYERS,
    )
    assert sorted(rows[0]["populations"]) == ["candidate", "superset"]


def test_out_of_bounds_position_raises(slug_dir):
    """An execution feature reaching the measurement set must fail loudly."""
    contexts, step_features, stats, rhyme_feats = build(slug_dir)
    execution = next(s for s in stats if s["feat_key"] == (2, 200))
    with pytest.raises(AssertionError, match="measurement length"):
        build_measurement_set(
            {"random_control": [execution]},
            contexts,
            collect_ctx_idx(step_features),
            RHYME_STEP,
            N_LAYERS,
        )


def test_matched_control_is_deterministic_and_uncapped(slug_dir):
    _, _, stats, rhyme_feats = build(slug_dir)
    pops = split_populations(stats, rhyme_feats, RHYME_STEP)
    a = sample_matched_control(pops["candidates"], pops["rest"], 0)
    b = sample_matched_control(pops["candidates"], pops["rest"], 0)
    assert [s["feat_key"] for s in a] == [s["feat_key"] for s in b]


# ------------------------------------------------------- last-layer exclusion


def test_feature_stats_drops_last_layer_features():
    """Layer n_layers-1 has no attention after it, so its features are unmeasurable.

    Attention precedes the MLP in a block, so nothing crosses positions after
    `blocks.{n_layers-1}.hook_mlp_out`. Suppressing such a feature at a position
    before the readout is bit-identical to baseline *by construction*. The
    pipeline used to admit these and pool the structural zeros as measured nulls
    -- 128 of 230 rows in the 270M/`inspire` measurement. See
    `methodology_evidence.md` section 9.
    """
    timeline = {(1, 100): {0: 0.5, 1: 0.4}, (5, 500): {0: 0.5, 1: 0.4}}
    percentiles = {(1, 100): {0: 90.0, 1: 80.0}, (5, 500): {0: 90.0, 1: 80.0}}

    keys = {s["feat_key"] for s in feature_stats(timeline, percentiles, 1, N_LAYERS)}
    assert keys == {(1, 100)}

    # Nothing special about 5: the cut tracks n_layers, not a constant.
    keys = {s["feat_key"] for s in feature_stats(timeline, percentiles, 1, 7)}
    assert keys == {(1, 100), (5, 500)}


def test_no_measured_row_is_in_the_last_layer(slug_dir):
    """The second guard: a population built without the filter must fail loudly.

    Distinct from the position checks above. Those catch a row at the wrong
    position; this catches a row at a position it can never influence, which
    would measure a guaranteed zero and report it as a null effect.
    """
    contexts, step_features, stats, _ = build(slug_dir)
    entry = next(s for s in stats if s["feat_key"] == (1, 100))
    unmeasurable = {**entry, "feat_key": (N_LAYERS - 1, 500)}

    with pytest.raises(RuntimeError, match="final layer"):
        build_measurement_set(
            {"candidate": [unmeasurable]},
            contexts,
            collect_ctx_idx(step_features),
            RHYME_STEP,
            N_LAYERS,
        )


def test_selection_percentile_is_computed_over_measurable_features_only():
    """Excluded features must not crowd survivors out of the candidate pool.

    Last-layer features have a direct path to the logit nodes, so they carry the
    highest attribution influence and sit at the top of every step's ranking.
    Ranking survivors against a distribution that still contains them pushes
    measurable features below the median, and `CANDIDATE_MIN_RHYME_PERCENTILE`
    then discards them. That is what collapsed 1B to a median of 2 candidates
    with two slugs at zero, while 4B -- proportionally less of it in the final
    layer -- was barely touched.

    Here three of four features are unmeasurable and outrank the survivor, so
    the survivor's all-nodes percentile is 25.0 (below the 50.0 cutoff) while
    its measurable percentile is 100.0 (it is the only one left).
    """
    timeline = {
        (1, 100): {0: 0.1, 1: 0.1},
        (5, 500): {0: 0.4, 1: 0.4},
        (5, 501): {0: 0.3, 1: 0.3},
        (5, 502): {0: 0.2, 1: 0.2},
    }
    percentiles = {
        (1, 100): {0: 25.0, 1: 25.0},
        (5, 500): {0: 100.0, 1: 100.0},
        (5, 501): {0: 75.0, 1: 75.0},
        (5, 502): {0: 50.0, 1: 50.0},
    }

    stats = feature_stats(timeline, percentiles, 1, N_LAYERS)
    survivor = next(s for s in stats if s["feat_key"] == (1, 100))

    # The descriptive percentile is untouched -- it still describes the graph.
    assert survivor["rhyme_percentile"] == 25.0
    assert survivor["rhyme_percentile"] < CANDIDATE_MIN_RHYME_PERCENTILE

    # The selection percentile ranks it among measurable features only.
    assert survivor["rhyme_percentile_measurable"] == 100.0

    # And selection uses the second, so the feature survives its own cutoff.
    survivor["peak_step"] = 0  # peaks strictly before the rhyme step
    picked = select_candidates([survivor], {(1, 100)})
    assert [s["feat_key"] for s in picked] == [(1, 100)]
