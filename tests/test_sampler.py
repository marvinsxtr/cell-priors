"""Tests for the grouped scale-free GRN sampler."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from cell_priors.samplers import GroupedScaleFreeSampler
from cell_priors.samplers.grouped_scale_free import grouped_scale_free_edges


def _real_edges(grn):
    """Regulator/target arrays for the real (non-padding, non-self-loop) edges."""
    real = np.asarray(grn.weight) > 0
    return np.asarray(grn.reg_idx)[real], np.asarray(grn.tar_idx)[real]


def test_deterministic_given_key():
    s = GroupedScaleFreeSampler(r=4.0, num_groups=2)
    a = s.sample(jax.random.PRNGKey(0), num_genes=50)
    b = s.sample(jax.random.PRNGKey(0), num_genes=50)
    assert np.array_equal(np.asarray(a.reg_idx), np.asarray(b.reg_idx))
    assert np.array_equal(np.asarray(a.tar_idx), np.asarray(b.tar_idx))
    assert np.array_equal(np.asarray(a.weight), np.asarray(b.weight))


def test_structure_is_valid():
    s = GroupedScaleFreeSampler(r=4.0, num_groups=3, kappa=10.0)
    grn = s.sample(jax.random.PRNGKey(1), num_genes=80)
    reg, tar = _real_edges(grn)
    assert grn.num_genes == 80
    assert len(reg) > 0
    assert not np.any(reg == tar)  # no self-loops among real edges
    assert reg.max() < 80 and tar.max() < 80


def test_scale_free_hubs():
    # A scale-free out-degree distribution: max out-degree >> mean out-degree.
    s = GroupedScaleFreeSampler(r=4.0, num_groups=1)
    grn = s.sample(jax.random.PRNGKey(2), num_genes=200)
    reg, _ = _real_edges(grn)
    outdeg = np.bincount(reg, minlength=200)
    assert outdeg.max() > 5 * max(outdeg.mean(), 1.0)


def test_groups_assigned():
    s = GroupedScaleFreeSampler(r=4.0, num_groups=4)
    grn = s.sample(jax.random.PRNGKey(3), num_genes=100)
    groups = np.unique(np.asarray(grn.group))
    assert len(groups) >= 2  # multiple modules present
    assert groups.min() >= 0 and groups.max() < 4


def test_padding_is_inert():
    # Duplicate / self-loop / padding slots carry zero weight; real edges carry their
    # pair multiplicity (a positive integer) on a single representative row.
    s = GroupedScaleFreeSampler(r=3.0, num_groups=2)
    grn = s.sample(jax.random.PRNGKey(4), num_genes=60)
    weight = np.asarray(grn.weight)
    reg, tar = np.asarray(grn.reg_idx), np.asarray(grn.tar_idx)
    assert np.all(weight >= 0)
    assert np.all(weight == np.round(weight))  # integer multiplicities
    assert np.all(weight[reg == tar] == 0.0)  # any self-loop slot is masked out


def test_jittable():
    fn = jax.jit(lambda key: grouped_scale_free_edges(
        key, 50, alpha=1e-6, beta=2 / 3 - 1e-6, gamma=1 / 3, delta_in=100.0, delta_out=1.0, k=2, kappa=10.0
    ))
    s, t, valid, groups = jax.block_until_ready(fn(jax.random.PRNGKey(0)))
    assert s.shape == t.shape == valid.shape
    assert groups.shape == (50,)


def test_vmap_over_keys():
    fn = jax.vmap(lambda key: grouped_scale_free_edges(
        key, 40, alpha=1e-6, beta=3 / 4 - 1e-6, gamma=1 / 4, delta_in=10.0, delta_out=1.0, k=3, kappa=5.0
    ))
    keys = jax.random.split(jax.random.PRNGKey(1), 8)
    s, t, valid, groups = fn(keys)
    assert s.shape[0] == 8 and groups.shape == (8, 40)
    assert jnp.all(valid.sum(axis=1) > 0)  # every sampled graph has real edges


def test_vmap_over_traced_hyperparams():
    # Per-network structure hyperparameters (gamma, kappa) may be traced as long as
    # n / k / max_edges stay static and an explicit max_edges is passed.
    n, k, max_edges = 40, 2, 220

    def one(key, gamma, kappa):
        return grouped_scale_free_edges(
            key, n, alpha=1e-6, beta=1.0 - 1e-6 - gamma, gamma=gamma,
            delta_in=50.0, delta_out=1.0, k=k, kappa=kappa, max_edges=max_edges,
        )

    b = 6
    keys = jax.random.split(jax.random.PRNGKey(0), b)
    gammas = jnp.linspace(0.2, 0.45, b)
    kappas = jnp.linspace(1.0, 12.0, b)
    s, t, valid, groups = jax.jit(jax.vmap(one))(keys, gammas, kappas)
    assert s.shape == (b, max_edges) and groups.shape == (b, n)
    assert jnp.all(valid.sum(axis=1) > 0)


def test_traced_hyperparams_require_explicit_max_edges():
    import pytest

    with pytest.raises(ValueError, match="max_edges"):
        jax.jit(lambda gamma: grouped_scale_free_edges(jax.random.PRNGKey(0), 30, gamma=gamma, beta=1.0 - gamma, alpha=0.0))(
            jnp.float32(0.3)
        )
