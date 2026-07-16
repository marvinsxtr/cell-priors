"""Host numpy reference for the canonical greedy DAG cycle removal.

Mirror of :func:`cell_priors.simulators.sergio.dag.dag_edge_mask`, written as a plain
loop so the jittable implementation can be validated against it exactly.
"""

from __future__ import annotations

import numpy as np


def _reachable(adj: np.ndarray) -> np.ndarray:
    """Length->=1 reachability of a boolean adjacency (exact transitive closure)."""
    reach = adj.astype(bool)
    g = adj.shape[-1]
    for _ in range(max(1, g)):
        nxt = reach | (reach.astype(np.int64) @ reach.astype(np.int64) > 0)
        if np.array_equal(nxt, reach):
            break
        reach = nxt
    return reach


def dag_edge_mask_reference(
    reg_idx: np.ndarray,
    tar_idx: np.ndarray,
    abs_weight: np.ndarray,
    active: np.ndarray,
    num_genes: int,
) -> np.ndarray:
    """Keep-mask selecting an acyclic subset of the active edges.

    Args:
        reg_idx: regulator gene index per edge, ``(E,)``.
        tar_idx: target gene index per edge, ``(E,)``.
        abs_weight: non-negative edge weight (interaction magnitude), ``(E,)``.
        active: initial per-edge active flag, ``(E,)``.
        num_genes: number of genes.

    Returns:
        Boolean ``(E,)`` mask, a subset of ``active`` whose kept edges form a DAG.
    """
    g = num_genes
    reg = np.asarray(reg_idx)
    tar = np.asarray(tar_idx)
    w = np.asarray(abs_weight, dtype=np.float64)
    keep = np.asarray(active) > 0

    while True:
        adj = np.zeros((g, g), dtype=np.float64)
        np.maximum.at(adj, (reg, tar), keep.astype(np.float64))
        reach = _reachable(adj)
        on_cycle = keep & (reach[tar, reg] > 0)
        if not on_cycle.any():
            break
        masked_w = np.where(on_cycle, w, np.inf)
        worst = int(np.argmin(masked_w))
        keep[worst] = False
    return keep
