"""Unit tests for the Hill production function."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from cell_priors.priors.sergio.core import _edge_production
from cell_priors.priors.sergio.grn import make_params


def _reference_hill(x_reg, h, n, k):
    """Direct transcription of sergio_rs Interaction::get_hill."""
    val = x_reg**n / (x_reg**n + h**n)
    return k * val if k > 0 else abs(k) * (1.0 - val)


def _with_h(p, h, **extra):
    return dataclasses.replace(p, h=jnp.asarray(h, dtype=p.h.dtype), **extra)


def test_activation_and_repression_match_reference():
    # Two edges into gene 2: an activator from 0 and a repressor from 1.
    reg = np.array([0, 1])
    tar = np.array([2, 2])
    k = np.array([3.0, -2.0])
    n = np.array([2.0, 2.0])
    p = _with_h(make_params(reg, tar, k, n, np.ones(3), np.ones((3, 1))), [1.5, 2.0])

    x = jnp.asarray([[1.2], [0.7], [0.0]])  # (G, C)
    prod = np.asarray(_edge_production(x, p))

    expected = _reference_hill(1.2, 1.5, 2.0, 3.0) + _reference_hill(0.7, 2.0, 2.0, -2.0)
    assert abs(prod[2, 0] - expected) < 1e-5
    assert prod[0, 0] == 0.0 and prod[1, 0] == 0.0  # genes with no in-edges


def test_repression_is_high_at_zero_regulator():
    # A pure repressor: at x_reg=0 the Hill term is 0 so contribution == |k|.
    p = _with_h(make_params(np.array([0]), np.array([1]), np.array([-4.0]), np.array([2.0]), np.ones(2), np.ones((2, 1))), [1.0])
    prod = np.asarray(_edge_production(jnp.asarray([[0.0], [0.0]]), p))
    assert abs(prod[1, 0] - 4.0) < 1e-6


def test_edge_mask_removes_contribution():
    p = make_params(np.array([0]), np.array([1]), np.array([3.0]), np.array([2.0]), np.ones(2), np.ones((2, 1)))
    p = _with_h(p, [1.0], edge_mask=jnp.zeros(1, dtype=p.edge_mask.dtype))
    prod = np.asarray(_edge_production(jnp.asarray([[2.0], [0.0]]), p))
    assert prod[1, 0] == 0.0
