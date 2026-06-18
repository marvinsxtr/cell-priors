"""Tests for the grouped scale-free GRN sampler."""

from __future__ import annotations

import jax
import numpy as np

from cell_priors.samplers import GroupedScaleFreeSampler


def test_deterministic_given_key():
    s = GroupedScaleFreeSampler(r=4.0, num_groups=2)
    a = s.sample(jax.random.PRNGKey(0), num_genes=50)
    b = s.sample(jax.random.PRNGKey(0), num_genes=50)
    assert np.array_equal(np.asarray(a.reg_idx), np.asarray(b.reg_idx))
    assert np.array_equal(np.asarray(a.tar_idx), np.asarray(b.tar_idx))


def test_structure_is_valid():
    s = GroupedScaleFreeSampler(r=4.0, num_groups=3, kappa=10.0)
    grn = s.sample(jax.random.PRNGKey(1), num_genes=80)
    reg, tar = np.asarray(grn.reg_idx), np.asarray(grn.tar_idx)
    assert grn.num_genes == 80
    assert grn.num_edges > 0
    assert not np.any(reg == tar)  # no self-loops
    assert reg.max() < 80 and tar.max() < 80


def test_scale_free_hubs():
    # A scale-free out-degree distribution: max out-degree >> mean out-degree.
    s = GroupedScaleFreeSampler(r=4.0, num_groups=1)
    grn = s.sample(jax.random.PRNGKey(2), num_genes=200)
    outdeg = np.bincount(np.asarray(grn.reg_idx), minlength=200)
    assert outdeg.max() > 5 * max(outdeg.mean(), 1.0)


def test_groups_assigned():
    s = GroupedScaleFreeSampler(r=4.0, num_groups=4)
    grn = s.sample(jax.random.PRNGKey(3), num_genes=100)
    groups = np.unique(np.asarray(grn.group))
    assert len(groups) >= 2  # multiple modules present
    assert groups.min() >= 0 and groups.max() < 4
