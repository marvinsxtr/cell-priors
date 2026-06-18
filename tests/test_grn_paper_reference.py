"""Numerically validate the JAX grn-paper core against the original numpy code.

The two use different RNGs, so we compare in the deterministic regime (``s = 0``),
where the SDE is a fixed ODE map and both converge to the same steady state. The
reference implementation in ``tests/reference/grn_paper_original.py`` is copied
from the grn-paper repository (MIT, (c) 2023 maguirre1).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from reference.grn_paper_original import simulate_rna_reference

from cell_priors.simulators.grn_paper.core import GrnPaperConfig, GrnPaperParams, simulate


def _random_dense(seed, g, density=0.3):
    rng = np.random.default_rng(seed)
    beta = rng.normal(0, 1, (g, g))
    mask = rng.random((g, g)) < density
    beta = beta * mask
    np.fill_diagonal(beta, 0.0)
    alpha = rng.normal(-1.0, 0.5, g)
    l = rng.uniform(0.3, 0.9, g)
    return beta, alpha, l


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_matches_reference_deterministic(seed):
    g = 10
    beta, alpha, l = _random_dense(seed, g)
    n_steps, burnin, dt = 2000, 1000, 1e-2

    # Reference (numpy); disable the convergence early-stop (step > tmax) so it runs
    # the full window, matching our fixed-length integration.
    ref = simulate_rna_reference(
        beta,
        alpha.reshape(-1, 1),
        l.reshape(-1, 1),
        s=0.0,
        dt=dt,
        tmax=n_steps,
        burnin=burnin,
        step=n_steps + 1,
    )[0]

    params = GrnPaperParams.from_dense(beta, alpha, l)
    cfg = GrnPaperConfig(num_cells=1, n_steps=n_steps, burnin=burnin, dt=dt, s=0.0)
    mine = np.asarray(simulate(params, jax.random.PRNGKey(0), cfg))[0]

    assert np.max(np.abs(mine - ref)) < 1e-3
