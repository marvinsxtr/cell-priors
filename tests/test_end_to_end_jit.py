"""The full prior -- structure sampling + simulation -- in a single jit / vmap.

With the sampler expressed as a scan, ``sample_params`` (draw a GRN structure, then
attach kinetics) is jittable, so a whole batch of distinct networks can be sampled and
simulated inside one compiled, vmapped graph -- no host-side network pool, no
device round-trips between sampling and simulation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from cell_priors import MapPFNPrior
from cell_priors.base import ComposedPrior, InterventionKind
from cell_priors.samplers import GroupedScaleFreeSampler
from cell_priors.simulators.grn_paper import GrnPaperConfig, GrnPaperSimulator
from cell_priors.simulators.sergio import SergioConfig


def _prior(num_cells: int = 16) -> ComposedPrior:
    sampler = GroupedScaleFreeSampler(r=3.0, num_groups=2, kappa=8.0)
    sim = GrnPaperSimulator(GrnPaperConfig(num_cells=num_cells, n_steps=400, burnin=200))
    return ComposedPrior(sampler, sim)


def test_sergio_prior_sample_and_simulate_in_one_jit():
    """The cycle-tolerant SERGIO prior (MapPFN) samples + simulates inside one jit/vmap."""
    prior = MapPFNPrior(SergioConfig(num_cells=12, num_cell_types=1, safety_iter=60, scale_iter=3))

    def run(key):
        k_struct, k_sim = jax.random.split(key)
        params = prior.sample_params(k_struct, num_genes=20)
        return prior.observational(params, k_sim), params.reg_idx

    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    expr, reg = jax.block_until_ready(jax.jit(jax.vmap(run))(keys))
    assert expr.shape == (4, 12, 20)
    assert not bool(jnp.isnan(expr).any()) and bool((expr >= 0).all())
    assert not np.array_equal(np.asarray(reg[0]), np.asarray(reg[1]))  # distinct networks


def test_sample_and_simulate_in_one_jit():
    prior = _prior()

    @jax.jit
    def run(key):
        k_struct, k_sim = jax.random.split(key)
        params = prior.sample_params(k_struct, num_genes=24)  # structure + kinetics
        return prior.observational(params, k_sim)

    expr = jax.block_until_ready(run(jax.random.PRNGKey(0)))
    assert expr.shape == (16, 24)
    assert not bool(jnp.isnan(expr).any())
    assert bool((expr >= 0).all())


def test_vmap_distinct_networks_end_to_end():
    """A batch of independent (structure + simulation) draws, fully vmapped."""
    prior = _prior(num_cells=8)

    def run(key):
        k_struct, k_sim = jax.random.split(key)
        params = prior.sample_params(k_struct, num_genes=20)
        return prior.observational(params, k_sim), params.reg_idx

    keys = jax.random.split(jax.random.PRNGKey(1), 6)
    expr, reg = jax.block_until_ready(jax.vmap(run)(keys))
    assert expr.shape == (6, 8, 20)
    assert bool((expr >= 0).all())
    # The networks differ across the batch (distinct structures, not one shared graph).
    assert not np.array_equal(np.asarray(reg[0]), np.asarray(reg[1]))


def test_intervention_in_one_jit():
    prior = _prior()

    @jax.jit
    def run(key):
        k_struct, k_sim = jax.random.split(key)
        params = prior.sample_params(k_struct, num_genes=18)
        return prior.interventional(params, k_sim, jnp.array([0]), kind=InterventionKind.KNOCKOUT)

    expr = jax.block_until_ready(run(jax.random.PRNGKey(2)))
    assert expr.shape == (16, 18)
    assert np.isfinite(np.asarray(expr)).all()
