"""Measurement-layer checks that need no model.

`_summarize_pass` is where the logit fields are computed, and the one property
worth pinning is internal consistency: `exp(logit - logsumexp)` must reproduce
the probability stored beside it. Taken over a top-k slice instead of the full
vocabulary -- an easy mistake, since every other recorded field is top-k -- the
logsumexp would not be the softmax denominator and the two would silently
disagree.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from downstream_effects_addon import _summarize_pass, measure_features  # noqa: E402


def test_logsumexp_is_over_the_full_vocabulary():
    torch.manual_seed(0)
    logits = torch.randn(5000)
    out = _summarize_pass(logits, token_id=17)
    assert math.isclose(math.exp(out["logit"] - out["logsumexp"]), out["prob"], rel_tol=1e-5)


def test_rank_and_top_k_agree():
    logits = torch.tensor([0.0, 5.0, 1.0, 3.0])
    out = _summarize_pass(logits, token_id=1)
    assert out["rank"] == 0
    assert out["top_ids"][:3] == [1, 3, 2]


def test_out_of_range_position_raises_rather_than_being_swallowed():
    """The per-feature except must not absorb a position bug.

    A position past the end of the sequence means position derivation is broken,
    not that one feature happens to be inert -- if it were caught it would show
    up only as a quiet shortfall in n_measured.
    """
    tokens = torch.arange(10)
    baseline = {
        "logit": 0.0,
        "logsumexp": 0.0,
        "prob": 0.5,
        "rank": 0,
        "entropy": 0.0,
        "top_ids": [],
        "top_logits": [],
        "top_probs": [],
    }
    rows = [{"layer": 0, "feat": 1, "position": 10}]
    with pytest.raises(IndexError, match="outside measurement sequence"):
        measure_features(model=None, tokens=tokens, baseline=baseline, rows=rows, token_id=0)
