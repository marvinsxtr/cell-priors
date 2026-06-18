"""Tests for the cycle-tolerant MapPFN prior (any GRN, no required master regulators)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from cell_priors import MapPfnPrior
from cell_priors.base import GRN, InterventionKind
from cell_priors.samplers import GroupedScaleFreeSampler
from cell_priors.simulators.sergio import SergioConfig, SergioSimulator


def _cyclic_grn(num_genes=3):
    """A pure cycle 0->1->...->0: every gene has an in-edge, so there are no MRs."""
    reg = jnp.arange(num_genes)
    tar = (jnp.arange(num_genes) + 1) % num_genes
    return GRN(reg_idx=reg, tar_idx=tar, weight=jnp.ones(num_genes), group=jnp.zeros(num_genes, jnp.int32))


def _prior(num_cells=30):
    cfg = SergioConfig(num_cells=num_cells, num_cell_types=1, safety_iter=80, scale_iter=3)
    return MapPfnPrior(cfg, sampler=GroupedScaleFreeSampler(r=3.0, num_groups=2))


def test_keeps_cycles_no_dagify():
    prior = _prior()
    grn = prior.sampler.sample(jax.random.PRNGKey(0), num_genes=40)
    # The MapPFN adapter does not remove edges (no DAGification): same edge count.
    params = prior.simulator.build_params(grn, jax.random.PRNGKey(1))
    assert int(params.num_edges) == int(grn.num_edges)


def test_mr_less_cycle_is_driven():
    # A network with no master regulators must still produce non-zero expression
    # under MapPFN (basal production for every gene), unlike standard SERGIO.
    grn = _cyclic_grn(3)
    permissive = SergioSimulator(
        SergioConfig(num_cells=20, safety_iter=60, scale_iter=3, require_mrs=False), acyclic=False
    )
    strict = SergioSimulator(SergioConfig(num_cells=20, safety_iter=60, scale_iter=3, require_mrs=True), acyclic=False)
    key = jax.random.PRNGKey(0)

    p_perm = permissive.build_params(grn, key)
    assert int(np.asarray(p_perm.mr_mask).sum()) == 0  # genuinely no master regulators
    expr_perm = permissive.simulate(p_perm, key)
    assert float(expr_perm.mean()) > 0.0
    assert not bool(jnp.isnan(expr_perm).any())

    expr_strict = strict.simulate(strict.build_params(grn, key), key)
    assert float(expr_strict.max()) == 0.0  # degenerate without MRs


def test_observational_health_and_jit():
    prior = _prior()
    p = prior.sample_params(jax.random.PRNGKey(1), num_genes=30)
    expr = prior.observational(p, jax.random.PRNGKey(2))
    assert expr.shape == (30, 30)
    assert not bool(jnp.isnan(expr).any()) and bool((expr >= 0).all())

    @jax.jit
    def step(params, key):
        pert = prior.intervene(params, jnp.array([0]), kind=InterventionKind.KNOCKDOWN, strength=0.5)
        return prior.observational(pert, key).sum()

    assert np.isfinite(float(step(p, jax.random.PRNGKey(3))))


def test_interventions_work():
    prior = _prior()
    p = prior.sample_params(jax.random.PRNGKey(1), num_genes=25)
    key = jax.random.PRNGKey(2)
    g = int(np.asarray(p.reg_idx)[0])
    ko = prior.interventional(p, key, jnp.array([g]), kind=InterventionKind.KNOCKOUT)
    assert float(ko[:, g].max()) == 0.0
