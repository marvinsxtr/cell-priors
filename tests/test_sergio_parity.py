"""Numerical parity tests of the JAX SERGIO core against sergio_rs.

The simulators use different RNGs, so we cannot match individual stochastic
samples. Instead we exploit the fact that with ``noise_s = 0`` the SERGIO SDE is
deterministic and converges to a fixed point. We:

1. run sergio_rs with ``noise_s=0`` and a long burn-in to reach steady state;
2. recover the master-regulator production rates from that state
   (``prod = x* * decay`` since MR fixed point is ``prod/decay``);
3. feed those exact rates into the JAX core and check every gene's converged
   state matches sergio_rs.

This validates the Hill function, the half-response/steady-state estimation and
the SDE integration end to end.
"""

from __future__ import annotations

import jax
import numpy as np

from cell_priors.priors.sergio import core
from cell_priors.priors.sergio.grn import SergioConfig, make_params

from conftest import build_matched_grn, sergio_converged

DT = 0.01
SAFETY = 4000
NCELL = 4
SCALE = 2


def _parity(edges, decay, hill_n, k, num_cell_types, mr_seed):
    grn, mr_profile, num_genes, reg_idx, tar_idx, k, hill_n, decay = build_matched_grn(
        edges, decay, hill_n, k, num_cell_types, mr_seed
    )
    sergio_ss = sergio_converged(grn, mr_profile, num_genes, num_cell_types, SAFETY, NCELL, SCALE, DT)

    # Recover MR production rates from the converged sergio state.
    tars = {t for _, t in edges}
    mr_genes = [i for i in range(num_genes) if i not in tars]
    prod_rates = np.zeros((num_genes, num_cell_types))
    for i in mr_genes:
        prod_rates[i] = sergio_ss[i] * decay[i]

    p = make_params(reg_idx, tar_idx, k, hill_n, decay, prod_rates)
    cfg = SergioConfig(num_cells=NCELL, num_cell_types=num_cell_types, safety_iter=SAFETY, scale_iter=SCALE, dt=DT, noise_s=0.0)
    p_init, ss = core.init_steady_state(p, cfg)
    traj = core.simulate_trajectory(p_init, ss, jax.random.PRNGKey(0), cfg)
    my_ss = np.asarray(traj[-1])

    # MR detection must agree with the structural definition.
    assert set(np.where(np.asarray(p_init.mr_mask) > 0)[0]) == set(mr_genes)
    return sergio_ss, my_ss


def test_parity_small_dag(small_dag):
    edges, decay, hill_n, k = small_dag
    sergio_ss, my_ss = _parity(edges, decay, hill_n, k, num_cell_types=2, mr_seed=7)
    assert np.max(np.abs(my_ss - sergio_ss)) < 1e-3


def test_parity_chain_with_repression():
    # Linear chain 0->1->2->3 mixing activation and repression.
    edges = [(0, 1), (1, 2), (2, 3)]
    decay = np.array([0.8, 0.6, 1.0, 0.7])
    hill_n = np.array([2, 2, 2])
    k = np.array([3.0, -2.0, 4.0])
    sergio_ss, my_ss = _parity(edges, decay, hill_n, k, num_cell_types=1, mr_seed=3)
    assert np.max(np.abs(my_ss - sergio_ss)) < 1e-3


def test_parity_multiple_seeds(small_dag):
    edges, decay, hill_n, k = small_dag
    for seed in (1, 11, 23):
        sergio_ss, my_ss = _parity(edges, decay, hill_n, k, num_cell_types=3, mr_seed=seed)
        assert np.max(np.abs(my_ss - sergio_ss)) < 1e-3, f"seed {seed}"
