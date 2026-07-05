"""Interface and intervention tests for the BoolODE simulator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cell_priors.base import ComposedPrior, InterventionKind
from cell_priors.samplers import GroupedScaleFreeSampler
from cell_priors.simulators.boolode import BoolodeConfig, BoolodeSimulator


@pytest.fixture
def prior():
    cfg = BoolodeConfig(num_cells=24, n_steps=1500, burnin=750, dt=0.01)
    return ComposedPrior(GroupedScaleFreeSampler(r=3.0, num_groups=1), BoolodeSimulator(cfg))


def _expressed_gene(prior, p, key) -> int:
    """The most highly expressed gene (guaranteed responsive to its own perturbation)."""
    return int(np.asarray(prior.observational(p, key).mean(axis=0)).argmax())


def _sole_activator_edge(p) -> tuple[int, int]:
    """A ``(regulator, target)`` where the regulator is the target's only activator.

    Removing that edge (hard knockout) leaves the target with no activator -> it turns
    constitutive; merely silencing the regulator (soft knockdown) leaves it off -- the
    sharpest hard-vs-soft contrast.
    """
    active = np.asarray(p.edge_mask) > 0
    is_act = np.asarray(p.is_act) > 0.5
    reg, tar = np.asarray(p.reg_idx), np.asarray(p.tar_idx)
    act_in = np.bincount(tar[active & is_act], minlength=p.num_genes)
    rep_in = np.bincount(tar[active & ~is_act], minlength=p.num_genes)
    for e in np.where(active & is_act)[0]:
        # Sole activator and no repressor: knockout -> constitutive, knockdown -> off.
        if act_in[tar[e]] == 1 and rep_in[tar[e]] == 0:
            return int(reg[e]), int(tar[e])
    raise AssertionError("no sole-activator edge found")


def test_observational_shape(prior):
    p = prior.sample_params(jax.random.PRNGKey(0), num_genes=15)
    expr = prior.observational(p, jax.random.PRNGKey(1))
    assert expr.shape == (prior.simulator.cfg.num_cells, 15)
    assert not bool(jnp.isnan(expr).any())
    assert bool((expr >= 0).all())


def test_knockout_silences_gene(prior):
    p = prior.sample_params(jax.random.PRNGKey(0), num_genes=15)
    g = _expressed_gene(prior, p, jax.random.PRNGKey(1))
    ko = prior.interventional(p, jax.random.PRNGKey(1), jnp.array([g]), kind=InterventionKind.KNOCKOUT)
    assert float(ko[:, g].max()) == 0.0


def test_knockdown_is_monotone_in_strength(prior):
    p = prior.sample_params(jax.random.PRNGKey(0), num_genes=15)
    key = jax.random.PRNGKey(1)
    g = _expressed_gene(prior, p, key)
    obs = float(prior.observational(p, key)[:, g].mean())
    kd_half = float(
        prior.interventional(p, key, jnp.array([g]), kind=InterventionKind.KNOCKDOWN, strength=0.5)[:, g].mean()
    )
    kd_full = float(
        prior.interventional(p, key, jnp.array([g]), kind=InterventionKind.KNOCKDOWN, strength=1.0)[:, g].mean()
    )
    assert kd_full <= kd_half <= obs + 1e-3
    assert kd_full < obs


def test_soft_and_hard_differ_downstream(prior):
    p = prior.sample_params(jax.random.PRNGKey(2), num_genes=20)
    g, downstream = _sole_activator_edge(p)
    key = jax.random.PRNGKey(3)
    ko = prior.interventional(p, key, jnp.array([g]), kind=InterventionKind.KNOCKOUT)
    kd = prior.interventional(p, key, jnp.array([g]), kind=InterventionKind.KNOCKDOWN, strength=1.0)
    assert abs(float(ko[:, downstream].mean()) - float(kd[:, downstream].mean())) > 1e-2


def test_jit_single_graph(prior):
    p = prior.sample_params(jax.random.PRNGKey(0), num_genes=12)

    @jax.jit
    def pipeline(params, key):
        pert = prior.intervene(params, jnp.array([0]), kind=InterventionKind.KNOCKOUT)
        return prior.observational(pert, key).sum()

    assert np.isfinite(float(pipeline(p, jax.random.PRNGKey(1))))


def test_vmap_over_seeds(prior):
    p = prior.sample_params(jax.random.PRNGKey(0), num_genes=12)
    keys = jax.random.split(jax.random.PRNGKey(5), 4)
    batched = jax.vmap(lambda k: prior.observational(p, k))(keys)
    assert batched.shape[0] == 4


def test_end_to_end_jit_sample_and_simulate(prior):
    """The whole prior -- structure sampling + kinetics + simulation -- in one jit/vmap."""

    def one(k):
        ks, ksim = jax.random.split(k)
        params = prior.sample_params(ks, num_genes=12)
        return prior.observational(params, ksim)

    out = jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(0), 3))
    assert out.shape == (3, prior.simulator.cfg.num_cells, 12)
    assert not bool(jnp.isnan(out).any())
