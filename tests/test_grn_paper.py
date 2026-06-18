"""Tests for the grn-paper sigmoid-SDE simulator, incl. parity with the reference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from cell_priors.base import ComposedPrior, InterventionKind
from cell_priors.samplers import GroupedScaleFreeSampler
from cell_priors.simulators.grn_paper import GrnPaperConfig, GrnPaperSimulator
from cell_priors.simulators.grn_paper.core import GrnPaperParams, knockout, simulate


def _reference_mean(beta, alpha, l, n_steps, burnin, dt):
    """Direct transcription of grn.simulate_rna with s=0 (deterministic)."""
    g = len(alpha)
    x = np.zeros(g)
    acc = np.zeros(g)
    for i in range(n_steps - 1):
        x = np.maximum(0.0, x + dt * (expit(alpha + x @ beta) - l * x))
        if i + 1 >= burnin:
            acc += x
    return acc / (n_steps - burnin)


def test_parity_with_reference_deterministic():
    rng = np.random.default_rng(0)
    g = 8
    beta = rng.normal(0, 1, (g, g))
    np.fill_diagonal(beta, 0)
    alpha = rng.normal(-1, 0.5, g)
    l = rng.uniform(0.3, 0.9, g)

    n_steps, burnin, dt = 1200, 600, 1e-2
    ref = _reference_mean(beta, alpha, l, n_steps, burnin, dt)

    p = GrnPaperParams(
        beta=jnp.asarray(beta, jnp.float32),
        alpha=jnp.asarray(alpha, jnp.float32),
        l=jnp.asarray(l, jnp.float32),
        group=jnp.zeros(g, jnp.int32),
    )
    cfg = GrnPaperConfig(num_cells=1, n_steps=n_steps, burnin=burnin, dt=dt, s=0.0)
    mine = np.asarray(simulate(p, jax.random.PRNGKey(0), cfg))[0]
    assert np.max(np.abs(mine - ref)) < 1e-3


def test_simulate_shape_and_health():
    sampler = GroupedScaleFreeSampler(r=3.0, num_groups=2)
    prior = ComposedPrior(sampler, GrnPaperSimulator(GrnPaperConfig(num_cells=16, n_steps=800, burnin=400)))
    p = prior.sample_params(jax.random.PRNGKey(0), num_genes=30)
    expr = prior.observational(p, jax.random.PRNGKey(1))
    assert expr.shape == (16, 30)
    assert not bool(jnp.isnan(expr).any())
    assert bool((expr >= 0).all())


def test_knockout_zeros_outgoing_edges():
    sampler = GroupedScaleFreeSampler(r=3.0, num_groups=1)
    sim = GrnPaperSimulator(GrnPaperConfig(num_cells=4, n_steps=200, burnin=100))
    p = ComposedPrior(sampler, sim).sample_params(jax.random.PRNGKey(0), num_genes=20)
    g = int(np.asarray(jnp.abs(p.beta).sum(axis=1)).argmax())  # a hub regulator
    ko = knockout(p, jnp.array([g]))
    assert float(jnp.abs(ko.beta[g]).sum()) == 0.0


def test_jittable_and_intervenable():
    sampler = GroupedScaleFreeSampler(r=3.0, num_groups=1)
    sim = GrnPaperSimulator(GrnPaperConfig(num_cells=4, n_steps=200, burnin=100))
    prior = ComposedPrior(sampler, sim)
    p = prior.sample_params(jax.random.PRNGKey(0), num_genes=15)

    @jax.jit
    def pipeline(params, key):
        pert = prior.intervene(params, jnp.array([0]), kind=InterventionKind.KNOCKDOWN, strength=0.5)
        return prior.observational(pert, key).sum()

    assert np.isfinite(float(pipeline(p, jax.random.PRNGKey(1))))
