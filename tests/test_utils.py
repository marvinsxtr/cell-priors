"""Tests for GRN inference and expression diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from cell_priors.utils import (
    assert_healthy,
    edge_auroc,
    edges_to_adjacency,
    infer_grn_correlation,
    infer_grn_regression,
    summarize,
)


def test_summarize_counts_problems():
    x = np.array([[1.0, 0.0], [np.nan, -2.0], [np.inf, 3.0]])
    s = summarize(x)
    assert s.n_nan == 1 and s.n_inf == 1 and s.n_negative == 1
    assert s.n_cells == 3 and s.n_genes == 2


def test_assert_healthy_raises():
    with pytest.raises(ValueError):
        assert_healthy(np.array([[np.nan]]))
    assert_healthy(np.array([[1.0, 2.0]]))  # ok


def test_inference_recovers_planted_edges():
    # Plant a clear causal chain: g0 -> g1 -> g2, strong linear signal.
    rng = np.random.default_rng(0)
    n = 500
    g0 = rng.normal(5, 1, n)
    g1 = 2.0 * g0 + rng.normal(0, 0.3, n)
    g2 = 1.5 * g1 + rng.normal(0, 0.3, n)
    expr = np.exp(np.stack([g0, g1, g2], axis=1)) - 1  # undo log1p later
    true = edges_to_adjacency([0, 1], [1, 2], 3)
    scores = infer_grn_regression(expr, alpha=0.001)
    auroc = edge_auroc(true, scores, directed=False)
    assert auroc > 0.7


def test_correlation_scores_shape():
    expr = np.abs(np.random.default_rng(1).normal(size=(50, 6)))
    s = infer_grn_correlation(expr)
    assert s.shape == (6, 6)
    assert np.allclose(np.diag(s), 0.0)
