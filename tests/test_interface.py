"""Tests for the uniform Prior interface and interventions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cell_priors.base import ComposedPrior, InterventionKind
from cell_priors.samplers import GroupedScaleFreeSampler
from cell_priors.simulators.sergio import SergioConfig, SergioSimulator


@pytest.fixture
def prior():
    cfg = SergioConfig(num_cells=30, num_cell_types=2, safety_iter=60, scale_iter=3, dt=0.01, noise_s=1.0)
    return ComposedPrior(GroupedScaleFreeSampler(r=3.0, num_groups=1), SergioSimulator(cfg))


def test_observational_shape(prior):
    p = prior.sample_params(jax.random.PRNGKey(0), num_genes=15)
    expr = prior.observational(p, jax.random.PRNGKey(1))
    cfg = prior.simulator.cfg
    assert expr.shape == (cfg.num_cells * cfg.num_cell_types, 15)
    assert not bool(jnp.isnan(expr).any())
    assert bool((expr >= 0).all())


def test_knockout_silences_gene(prior):
    p = prior.sample_params(jax.random.PRNGKey(0), num_genes=15)
    g = int(np.where(np.asarray(p.mr_mask) > 0)[0][0])
    ko = prior.interventional(p, jax.random.PRNGKey(1), jnp.array([g]), kind=InterventionKind.KNOCKOUT)
    assert float(ko[:, g].max()) == 0.0


def test_knockdown_is_monotone_in_strength(prior):
    p = prior.sample_params(jax.random.PRNGKey(0), num_genes=15)
    g = int(np.where(np.asarray(p.mr_mask) > 0)[0][0])
    key = jax.random.PRNGKey(1)
    obs = float(prior.observational(p, key)[:, g].mean())
    kd_half = float(
        prior.interventional(p, key, jnp.array([g]), kind=InterventionKind.KNOCKDOWN, strength=0.5)[:, g].mean()
    )
    kd_full = float(
        prior.interventional(p, key, jnp.array([g]), kind=InterventionKind.KNOCKDOWN, strength=1.0)[:, g].mean()
    )
    # Stronger knockdown => lower expression of the targeted gene.
    assert kd_full <= kd_half <= obs + 1e-3
    assert kd_full < obs


def test_soft_and_hard_differ_downstream(prior):
    # Knockdown keeps the regulatory edge; knockout removes it, so downstream
    # genes should generally respond differently.
    p = prior.sample_params(jax.random.PRNGKey(2), num_genes=20)
    reg = np.asarray(p.reg_idx)
    tar = np.asarray(p.tar_idx)
    g = int(reg[0])
    downstream = int(tar[0])
    key = jax.random.PRNGKey(3)
    ko = prior.interventional(p, key, jnp.array([g]), kind=InterventionKind.KNOCKOUT)
    kd = prior.interventional(p, key, jnp.array([g]), kind=InterventionKind.KNOCKDOWN, strength=1.0)
    assert abs(float(ko[:, downstream].mean()) - float(kd[:, downstream].mean())) > 1e-3


def test_jit_single_graph(prior):
    p = prior.sample_params(jax.random.PRNGKey(0), num_genes=12)

    @jax.jit
    def pipeline(params, key):
        pert = prior.intervene(params, jnp.array([0]), kind=InterventionKind.KNOCKOUT)
        return prior.observational(pert, key).sum()

    out = float(pipeline(p, jax.random.PRNGKey(1)))
    assert np.isfinite(out)


def test_vmap_over_seeds(prior):
    p = prior.sample_params(jax.random.PRNGKey(0), num_genes=12)
    keys = jax.random.split(jax.random.PRNGKey(5), 4)
    batched = jax.vmap(lambda k: prior.observational(p, k))(keys)
    assert batched.shape[0] == 4
