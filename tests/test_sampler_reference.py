"""Numerically validate the GRN representation against the grn-paper reference.

The interesting failure mode is not the (RNG-dependent) growth process but how the
sampled directed multigraph is turned into a :class:`GRN` -- in particular how parallel
edges are counted. grn-paper builds the interaction matrix by *summing* parallel edges
(``β = S ⊙ E`` with ``E`` the multiplicity adjacency). These tests pin that down exactly:
for real reference multigraphs (which contain parallel edges), the GRN's reconstructed
adjacency must equal the reference multiplicity adjacency. The reference in
``tests/reference/grouped_scale_free_original.py`` is copied from the grn-paper repository
(MIT, (c) 2023 maguirre1).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from reference.grouped_scale_free_original import grouped_scale_free_edges_reference

from cell_priors.samplers.grouped_scale_free import edges_to_grn

# beta-heavy regime so the reference produces plenty of parallel edges.
_PARAMS = {
    "alpha": 1e-6,
    "beta": 1.0 - 1e-6 - 0.4,
    "gamma": 0.4,
    "delta_in": 50.0,
    "delta_out": 1.0,
    "k": 2,
    "kappa": 8.0,
}


def _adjacency(reg: np.ndarray, tar: np.ndarray, weight: np.ndarray, n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float64)
    np.add.at(adj, (reg, tar), weight)
    return adj


@pytest.mark.parametrize("seed", [0, 1, 7, 13])
def test_grn_multiplicity_matches_reference(seed):
    n = 40
    s, t, groups = grouped_scale_free_edges_reference(n, seed=seed, **_PARAMS)

    # grn-paper's interaction adjacency: parallel edges summed, self-loops dropped.
    keep = s != t
    ref_adj = _adjacency(s[keep], t[keep], np.ones(keep.sum()), n)
    assert (ref_adj > 1).any(), "reference graph has no parallel edges -- test is not exercising multiplicity"

    # Our GRN representation, built from the same raw multigraph (self-loops marked invalid).
    valid = jnp.asarray((s != t).astype(np.float32))
    grn = edges_to_grn(jnp.asarray(s), jnp.asarray(t), valid, jnp.asarray(groups))
    our_adj = _adjacency(np.asarray(grn.reg_idx), np.asarray(grn.tar_idx), np.asarray(grn.weight), n)

    assert np.array_equal(our_adj, ref_adj)


def test_edges_to_grn_collapses_parallel_edges():
    # A hand-built multigraph: (0->1) x3, (1->2) x1, (2->3) x2, plus a self-loop (3->3).
    sources = jnp.array([0, 0, 0, 1, 2, 2, 3])
    targets = jnp.array([1, 1, 1, 2, 3, 3, 3])
    valid = jnp.asarray((np.asarray(sources) != np.asarray(targets)).astype(np.float32))
    groups = jnp.zeros(4, jnp.int32)

    grn = edges_to_grn(sources, targets, valid, groups)
    adj = _adjacency(np.asarray(grn.reg_idx), np.asarray(grn.tar_idx), np.asarray(grn.weight), 4)

    assert adj[0, 1] == 3 and adj[1, 2] == 1 and adj[2, 3] == 2
    assert adj[3, 3] == 0  # self-loop dropped
    assert adj.sum() == 6  # 3 + 1 + 2, total multiplicity preserved
    # The multiplicity sits on exactly one representative row per unique pair.
    assert int((np.asarray(grn.weight) > 0).sum()) == 3
