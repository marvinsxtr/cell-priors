"""Shared test fixtures and helpers for building matched SERGIO networks."""

from __future__ import annotations

import numpy as np
import pytest
import sergio_rs


def gene_name(i: int) -> str:
    """Match the gene naming used so sergio_rs sorted order == index order."""
    return f"GENE{i:04d}"


def build_matched_grn(edges, decay, hill_n, k, num_cell_types, mr_seed):
    """Build a sergio_rs GRN and the matching JAX inputs from an edge list.

    ``edges`` is a list of ``(reg_idx, tar_idx)`` tuples. ``decay`` is per gene;
    ``hill_n`` and ``k`` are per edge. Returns ``(sergio_grn, mr_profile, num_genes,
    reg_idx, tar_idx, k, hill_n, decay)``.
    """
    num_genes = max(max(r, t) for r, t in edges) + 1
    grn = sergio_rs.GRN()
    for (r, t), kk, nn in zip(edges, k, hill_n):
        grn.add_interaction(
            reg=sergio_rs.Gene(gene_name(r), float(decay[r])),
            tar=sergio_rs.Gene(gene_name(t), float(decay[t])),
            k=float(kk),
            h=None,
            n=int(nn),
        )
    grn.set_mrs()
    mr_profile = sergio_rs.MrProfile.from_random(
        grn, num_cell_types=num_cell_types, low_range=(0.5, 2.0), high_range=(3.0, 5.0), seed=mr_seed
    )
    reg_idx = np.array([r for r, _ in edges])
    tar_idx = np.array([t for _, t in edges])
    return (
        grn,
        mr_profile,
        num_genes,
        reg_idx,
        tar_idx,
        np.asarray(k),
        np.asarray(hill_n, dtype=float),
        np.asarray(decay),
    )


def sergio_converged(grn, mr_profile, num_genes, num_cell_types, safety_iter, num_cells, scale_iter, dt):
    """Run sergio_rs with ``noise_s=0`` and return the converged ``(G, C)`` state."""
    sim = sergio_rs.Sim(
        grn,
        num_cells=num_cells,
        noise_s=0.0,
        safety_iter=safety_iter,
        scale_iter=scale_iter,
        dt=dt,
        seed=1,
    )
    df = sim.simulate(mr_profile)
    data = df.drop("Genes").to_numpy()  # (G, C * num_cells)
    return data.reshape(num_genes, num_cell_types, num_cells)[:, :, 0]  # (G, C)


@pytest.fixture
def small_dag():
    """A fixed small DAG with two master regulators and a diamond."""
    edges = [(0, 2), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (1, 4)]
    num_genes = 6
    decay = np.full(num_genes, 0.8)
    hill_n = np.full(len(edges), 2)
    k = np.full(len(edges), 3.0)
    return edges, decay, hill_n, k
