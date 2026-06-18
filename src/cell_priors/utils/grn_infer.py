"""Basic GRN inference / causal discovery from expression data.

These are intentionally simple, dependency-light baselines, useful for (a)
generating *comparable* synthetic data -- infer a GRN from a real dataset, then
re-simulate it with the SERGIO prior -- and (b) sanity-checking / overfitting
experiments where you want a controllable signal-to-noise knob.

Two scorers are provided: marginal absolute correlation and a GENIE3-style
per-target regression (each gene regressed on all others). Both return a dense
``(num_genes, num_genes)`` matrix of edge scores ``S[reg, tar]``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def infer_grn_correlation(expr: NDArray, log1p: bool = True) -> NDArray:
    """Score edges by absolute Pearson correlation between genes.

    ``expr`` is ``(num_cells, num_genes)``. Returns a symmetric score matrix with
    a zero diagonal.
    """
    x = np.log1p(expr) if log1p else np.asarray(expr, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    x = x / std
    n = x.shape[0]
    corr = (x.T @ x) / max(n - 1, 1)
    scores = np.abs(corr)
    np.fill_diagonal(scores, 0.0)
    return scores


def infer_grn_regression(expr: NDArray, alpha: float = 0.01, log1p: bool = True) -> NDArray:
    """GENIE3-style inference: regress each target gene on all other genes.

    Uses Lasso so the coefficients are sparse. ``S[reg, tar]`` is ``|coef|`` of
    regulator ``reg`` when predicting target ``tar``.
    """
    from sklearn.linear_model import Lasso

    x = np.log1p(expr) if log1p else np.asarray(expr, dtype=float)
    num_genes = x.shape[1]
    scores = np.zeros((num_genes, num_genes))
    for tar in range(num_genes):
        mask = np.arange(num_genes) != tar
        model = Lasso(alpha=alpha, max_iter=2000)
        model.fit(x[:, mask], x[:, tar])
        scores[mask, tar] = np.abs(model.coef_)
    return scores


def edges_to_adjacency(reg_idx: NDArray, tar_idx: NDArray, num_genes: int) -> NDArray:
    """Build a 0/1 directed adjacency matrix from an edge list."""
    adj = np.zeros((num_genes, num_genes))
    adj[np.asarray(reg_idx), np.asarray(tar_idx)] = 1.0
    return adj


def edge_auroc(true_adj: NDArray, scores: NDArray, directed: bool = True) -> float:
    """AUROC of recovering ground-truth edges from a score matrix.

    The diagonal is excluded. If ``directed`` is False the comparison is made on
    the symmetrized graph (correct for undirected scorers like correlation).
    """
    from sklearn.metrics import roc_auc_score

    true = np.asarray(true_adj).copy()
    s = np.asarray(scores).copy()
    if not directed:
        true = ((true + true.T) > 0).astype(float)
        s = np.maximum(s, s.T)
    off = ~np.eye(true.shape[0], dtype=bool)
    y_true = (true[off] > 0).astype(int)
    y_score = s[off]
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))
