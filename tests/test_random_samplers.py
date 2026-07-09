"""Tests for the scale-free, Erdős–Rényi and Watts–Strogatz GRN samplers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from cell_priors.samplers import (
    ErdosRenyiSampler,
    ScaleFreeSampler,
    WattsStrogatzSampler,
    erdos_renyi_edges,
    scale_free_edges,
    watts_strogatz_edges,
)


def _real_edges(grn):
    """Regulator/target arrays for the real (non-padding, non-self-loop) edges."""
    real = np.asarray(grn.weight) > 0
    return np.asarray(grn.reg_idx)[real], np.asarray(grn.tar_idx)[real]


def test_deterministic_given_key():
    for s in (ScaleFreeSampler(), ErdosRenyiSampler(), WattsStrogatzSampler()):
        a = s.sample(jax.random.PRNGKey(0), num_genes=50)
        b = s.sample(jax.random.PRNGKey(0), num_genes=50)
        assert np.array_equal(np.asarray(a.reg_idx), np.asarray(b.reg_idx))
        assert np.array_equal(np.asarray(a.tar_idx), np.asarray(b.tar_idx))


def test_structure_is_valid():
    for s in (ScaleFreeSampler(), ErdosRenyiSampler(), WattsStrogatzSampler()):
        grn = s.sample(jax.random.PRNGKey(1), num_genes=80)
        reg, tar = _real_edges(grn)
        assert grn.num_genes == 80
        assert len(reg) > 0
        assert not np.any(reg == tar)  # no self-loops among real edges
        assert reg.max() < 80 and tar.max() < 80
        assert np.array_equal(np.unique(np.asarray(grn.group)), np.array([0]))  # single module


def test_scale_free_has_out_degree_hubs():
    # Preferential attachment on out-degree: max out-degree >> mean out-degree.
    grn = ScaleFreeSampler(m=2).sample(jax.random.PRNGKey(2), num_genes=200)
    reg, _ = _real_edges(grn)
    outdeg = np.bincount(reg, minlength=200)
    assert outdeg.max() > 5 * max(outdeg.mean(), 1.0)


def test_erdos_renyi_is_homogeneous():
    # No hubs: the max out-degree stays within a small multiple of the mean.
    grn = ErdosRenyiSampler(mean_degree=3.0).sample(jax.random.PRNGKey(2), num_genes=200, max_edges=1000)
    reg, _ = _real_edges(grn)
    outdeg = np.bincount(reg, minlength=200)
    assert outdeg.max() < 5 * max(outdeg.mean(), 1.0)


def test_watts_strogatz_lattice_is_regular():
    # With no rewiring every node has out-degree exactly k_neighbors (a pure ring lattice).
    s, t, valid, _ = watts_strogatz_edges(jax.random.PRNGKey(0), n=60, k_neighbors=3, beta=0.0)
    outdeg = np.bincount(np.asarray(s)[np.asarray(valid) > 0], minlength=60)
    assert np.all(outdeg == 3)


def test_erdos_renyi_density_tracks_mean_degree():
    lo = int(np.asarray(erdos_renyi_edges(jax.random.PRNGKey(0), 100, 1.0, 800)[2]).sum())
    hi = int(np.asarray(erdos_renyi_edges(jax.random.PRNGKey(0), 100, 5.0, 800)[2]).sum())
    assert hi > lo


def test_jittable():
    n = 50
    fns = [
        lambda key: scale_free_edges(key, n, m=2, delta=1.0),
        lambda key: erdos_renyi_edges(key, n, mean_degree=2.0, max_edges=200),
        lambda key: watts_strogatz_edges(key, n, k_neighbors=2, beta=0.1),
    ]
    for fn in fns:
        s, t, valid, groups = jax.block_until_ready(jax.jit(fn)(jax.random.PRNGKey(0)))
        assert s.shape == t.shape == valid.shape
        assert groups.shape == (n,)


def test_vmap_over_traced_hyperparams():
    # The density / attachment / rewiring knobs may be traced (n and edge count stay static),
    # so a prior can vary structure per network under a single vmap.
    n, b = 40, 6
    keys = jax.random.split(jax.random.PRNGKey(1), b)

    s, t, valid, groups = jax.jit(jax.vmap(lambda k, d: scale_free_edges(k, n, 2, d)))(keys, jnp.linspace(0.5, 2.0, b))
    assert s.shape[0] == b and groups.shape == (b, n) and jnp.all(valid.sum(1) > 0)

    s, t, valid, groups = jax.jit(jax.vmap(lambda k, md: erdos_renyi_edges(k, n, md, 200)))(
        keys, jnp.linspace(1.5, 3.0, b)
    )
    assert s.shape == (b, 200) and jnp.all(valid.sum(1) > 0)

    s, t, valid, groups = jax.jit(jax.vmap(lambda k, be: watts_strogatz_edges(k, n, 2, be)))(
        keys, jnp.linspace(0.0, 0.5, b)
    )
    assert groups.shape == (b, n) and jnp.all(valid.sum(1) > 0)
