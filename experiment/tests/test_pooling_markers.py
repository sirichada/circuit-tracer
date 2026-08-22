"""The pooling guard in `comparing.py`.

Two code changes altered what a "candidate" means without altering the shape of
the result files: `9684989` (drop causally-unreachable last-layer features) and
`de8e67f` (rank the percentile cutoff over the measurable population). Files
from between them carry `excludes_last_layer: true` and look poolable.

Every statistic in `comparing.py` aggregates rows across prompts and sizes, so
mixing generations produces a number that looks fine and means nothing. These
pin the guard that refuses it.
"""

from __future__ import annotations

import pytest

from comparing import check_pooling_markers


def payload(excludes: bool | None = True, population: str | None = "measurable") -> dict:
    cfg: dict = {}
    if excludes is not None:
        cfg["excludes_last_layer"] = excludes
    if population is not None:
        cfg["selection_percentile_population"] = population
    return {"config": cfg}


def test_current_generation_pools_cleanly():
    results = {("270m", "realm"): payload(), ("4b", "realm"): payload()}
    check_pooling_markers(results)  # must not raise


def test_mixing_generations_raises():
    """The trap: both files claim excludes_last_layer, one is still stale."""
    results = {
        ("270m", "realm"): payload(),
        ("1b", "realm"): payload(population=None),  # 9684989..de8e67f
    }
    with pytest.raises(RuntimeError, match="disagree on how they were measured"):
        check_pooling_markers(results)


def test_pre_fix_files_are_not_silently_mixed_in():
    results = {
        ("270m", "realm"): payload(),
        ("1b", "realm"): payload(excludes=None, population=None),
    }
    with pytest.raises(RuntimeError, match="disagree"):
        check_pooling_markers(results)


def test_a_consistently_stale_pool_warns_but_runs(capsys):
    """Internally coherent old files are poolable with each other, just outdated.

    Raising here would block re-analysis of an archived run, which is a
    different situation from mixing two definitions in one aggregate.
    """
    results = {
        ("270m", "realm"): payload(population=None),
        ("1b", "realm"): payload(population=None),
    }
    check_pooling_markers(results)
    assert "stale" in capsys.readouterr().out


def test_absent_marker_means_old_not_unknown():
    """A missing key is evidence about the file, not a reason to skip the check.

    Treating absent as unknown-and-therefore-fine is how the first marker failed
    to catch anything.
    """
    results = {
        ("270m", "a"): payload(population="all_nodes"),
        ("270m", "b"): payload(population=None),
    }
    check_pooling_markers(results)  # both are all_nodes: consistent, no raise
