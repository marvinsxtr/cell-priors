"""Numerically validate the JAX BoolODE core against a reference implementation.

The two use different RNGs, so we compare in the deterministic regime (no noise),
where the SDE reduces to an ODE that converges to a fixed point. The reference in
``tests/reference/boolode_original.py`` builds BoolODE's transcription function the
documented way -- enumerating regulator combinations and evaluating the Boolean
rule -- while the JAX core computes the equivalent ``O(E)`` closed form. Agreement
across random signed DAGs validates that reimplementation end to end (Hill terms,
the activation function and the two-species integration together).
"""

from __future__ import annotations

import numpy as np
import pytest
from reference.boolode_original import boolode_steady_state

from cell_priors.simulators.boolode.core import BoolodeConfig, steady_state
from cell_priors.simulators.boolode.simulator import make_params

# Deterministic integration long enough for both integrators to reach steady state.
CFG = BoolodeConfig(num_cells=1, n_steps=8000, burnin=1, dt=0.01)


def _random_dag(seed, num_genes=6, avg_reg=2.0, repression=0.4):
    """A random DAG (edges low->high index, so gene 0 is a source) with signs + kinetics."""
    rng = np.random.default_rng(seed)
    reg, tar = [], []
    for j in range(1, num_genes):
        n = min(j, 1 + rng.poisson(avg_reg - 1))
        for rr in rng.choice(j, size=n, replace=False):
            reg.append(int(rr))
            tar.append(j)
    reg, tar = np.array(reg), np.array(tar)
    is_act = (rng.random(len(reg)) > repression).astype(float)
    thr = rng.uniform(9.0, 11.0, len(reg))
    hill_n = rng.uniform(9.0, 11.0, len(reg))
    m = rng.uniform(18.0, 22.0, num_genes)
    l_x = rng.uniform(9.0, 11.0, num_genes)
    r = rng.uniform(9.0, 11.0, num_genes)
    l_p = rng.uniform(0.9, 1.1, num_genes)
    return num_genes, reg, tar, is_act, thr, hill_n, m, l_x, r, l_p


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 11, 23])
def test_matches_reference_deterministic(seed):
    g, reg, tar, is_act, thr, hill_n, m, l_x, r, l_p = _random_dag(seed)
    ref = boolode_steady_state(g, reg, tar, is_act, thr, hill_n, m, l_x, r, l_p)

    p = make_params(reg, tar, is_act, thr, hill_n, m, l_x, r, l_p)
    x, _ = steady_state(p, CFG)
    mine = np.asarray(x)

    assert np.max(np.abs(mine - ref)) < 1e-3, f"seed {seed}: {np.abs(mine - ref)}"


@pytest.mark.parametrize("repression", [0.0, 1.0])
def test_matches_reference_pure_signs(repression):
    """All-activator and all-repressor networks (edge cases of the closed form)."""
    g, reg, tar, is_act, thr, hill_n, m, l_x, r, l_p = _random_dag(7, repression=repression)
    ref = boolode_steady_state(g, reg, tar, is_act, thr, hill_n, m, l_x, r, l_p)
    p = make_params(reg, tar, is_act, thr, hill_n, m, l_x, r, l_p)
    x, _ = steady_state(p, CFG)
    assert np.max(np.abs(np.asarray(x) - ref)) < 1e-3
