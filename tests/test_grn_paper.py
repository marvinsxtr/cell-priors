"""Tests for the grn-paper sigmoid-SDE simulator.

Numerical parity against the original numpy code lives in
``test_grn_paper_reference.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from cell_priors.base import ComposedPrior, InterventionKind
from cell_priors.samplers import GroupedScaleFreeSampler
from cell_priors.simulators.grn_paper import GrnPaperConfig, GrnPaperSimulator
from cell_priors.simulators.grn_paper.core import knockout


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
    reg = np.asarray(p.reg_idx)
    g = int(np.bincount(reg, minlength=p.num_genes).argmax())  # a hub regulator
    ko = knockout(p, jnp.array([g]))
    out_edges = np.asarray(ko.reg_idx) == g
    assert float(jnp.abs(jnp.asarray(ko.beta)[out_edges]).sum()) == 0.0


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
