"""Tests for the technical-noise pipeline."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cell_priors.priors.sergio.noise import DS_PROFILES, add_technical_noise, add_technical_noise_by_name


def _expr():
    return jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (100, 20))) * 50.0


def test_counts_are_nonnegative_integers():
    out = add_technical_noise_by_name(jax.random.PRNGKey(1), _expr(), "DS6")
    assert bool((out >= 0).all())
    assert bool(jnp.all(out == jnp.round(out)))


def test_dropout_introduces_sparsity():
    out = add_technical_noise_by_name(jax.random.PRNGKey(1), _expr(), "DS6")
    assert 0.0 < float((out == 0).mean()) < 1.0


def test_deterministic_given_key():
    e = _expr()
    a = add_technical_noise_by_name(jax.random.PRNGKey(1), e, "DS6")
    b = add_technical_noise_by_name(jax.random.PRNGKey(1), e, "DS6")
    assert bool(jnp.all(a == b))


def test_all_profiles_run():
    e = _expr()
    for name, profile in DS_PROFILES.items():
        out = add_technical_noise(jax.random.PRNGKey(2), e, profile)
        assert not bool(jnp.isnan(out).any()), name


def test_jittable():
    f = jax.jit(lambda k, e: add_technical_noise(k, e, DS_PROFILES["DS6"]))
    out = f(jax.random.PRNGKey(3), _expr())
    assert out.shape == (100, 20)
