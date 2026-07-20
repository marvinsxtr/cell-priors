"""Tests for the uniform Prior interface and interventions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cell_priors.base import ComposedPrior, InterventionKind
from cell_priors.samplers import GroupedScaleFreeSampler
from cell_priors.simulators.sergio import SergioConfig, SergioSimulator, core, make_params
from cell_priors.simulators.sergio.interventions import knockout


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


def test_knockout_orphan_collapses_not_promoted():
    # Chain 0 -> 1 -> 2 with gene 0 the only master regulator. Knocking out gene 1 orphans gene 2.
    # Matching sergio_rs, the MR set is fixed at build time: gene 2 must NOT be promoted to a master
    # regulator, so in strict mode (require_mrs) it collapses instead of settling at a basal rate.
    p = make_params(
        reg_idx=np.array([0, 1]),
        tar_idx=np.array([1, 2]),
        k=np.array([3.0, 3.0]),
        hill_n=np.array([2.0, 2.0]),
        decay=np.array([1.0, 1.0, 1.0]),
        prod_rates=np.array([[4.0], [4.0], [4.0]]),
    )
    cfg = SergioConfig(num_cells=200, require_mrs=True, safety_iter=120, scale_iter=8, dt=0.01)
    ko = knockout(p, jnp.array([1]))
    assert float(ko.mr_mask[2]) == 0.0, "orphaned target must not be promoted to a master regulator"

    key = jax.random.PRNGKey(0)
    ctrl_g2 = float(core.simulate(p, key, cfg)[:, 2].mean())
    ko_g2 = float(core.simulate(ko, key, cfg)[:, 2].mean())
    assert ctrl_g2 > 0.5, "gene 2 should be expressed in the control"
    assert ko_g2 < 0.1 * ctrl_g2, "orphaned gene 2 should collapse toward zero after the knockout"


def test_soft_and_hard_differ_downstream(prior):
    # A partial knockdown keeps the regulator partly active (and its edge intact) while a knockout
    # removes it entirely, so a downstream gene should respond differently.
    p = prior.sample_params(jax.random.PRNGKey(2), num_genes=20)
    reg = np.asarray(p.reg_idx)
    tar = np.asarray(p.tar_idx)
    g = int(reg[0])
    downstream = int(tar[0])
    key = jax.random.PRNGKey(3)
    ko = prior.interventional(p, key, jnp.array([g]), kind=InterventionKind.KNOCKOUT)
    kd = prior.interventional(p, key, jnp.array([g]), kind=InterventionKind.KNOCKDOWN, strength=0.5)
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
