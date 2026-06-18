"""Technical noise model in JAX (SERGIO ``add_technical_noise`` port).

Mirrors ``sergio_rs``'s noise pipeline -- outlier genes, cell library size,
logistic dropout and UMI count sampling -- but operates on a ``(num_cells,
num_genes)`` matrix (the layout the rest of this library uses) and is fully
jittable. The SERGIO reference operates on the transposed ``(genes, cells)``
matrix; the axes below are flipped accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, random


@dataclass(frozen=True)
class NoiseProfile:
    """Technical-noise parameters (the SERGIO paper's DS1-DS14 presets)."""

    outlier_mu: float
    library_mu: float
    library_sigma: float
    dropout_k: float
    dropout_q: float
    outlier_p: float = 0.01
    outlier_sigma: float = 1.0


# SERGIO paper dataset profiles, from sergio_rs/src/noise.rs.
DS_PROFILES: dict[str, NoiseProfile] = {
    "DS1": NoiseProfile(0.8, 4.8, 0.3, 20.0, 82.0),
    "DS2": NoiseProfile(0.8, 6.0, 0.4, 12.0, 80.0),
    "DS3": NoiseProfile(0.8, 7.0, 0.4, 8.0, 80.0),
    "DS4": NoiseProfile(3.0, 6.0, 0.3, 8.0, 74.0),
    "DS5": NoiseProfile(3.0, 6.0, 0.4, 8.0, 82.0),
    "DS6": NoiseProfile(5.0, 4.5, 0.7, 8.0, 45.0),
    "DS7": NoiseProfile(3.0, 4.4, 0.8, 8.0, 85.0),
    "DS8": NoiseProfile(4.5, 10.8, 0.55, 2.0, 92.0),
    "DS13": NoiseProfile(0.8, 3.6, 0.4, 8.0, 70.0),
    "DS14": NoiseProfile(0.8, 5.0, 0.4, 4.0, 80.0),
}


def _lognormal(key: Array, shape, mu: float, sigma: float) -> Array:
    return jnp.exp(random.normal(key, shape) * sigma + mu)


def add_outlier_effect(key: Array, expr: Array, p: float, mu: float, sigma: float) -> Array:
    """Scale a random ``p`` fraction of *genes* by a log-normal outlier factor."""
    n_genes = expr.shape[1]
    k_ind, k_fac = random.split(key)
    indicator = random.bernoulli(k_ind, p, (n_genes,))
    factors = _lognormal(k_fac, (n_genes,), mu, sigma)
    gene_scale = jnp.where(indicator, factors, 1.0)  # (G,)
    return expr * gene_scale[None, :]


def add_library_size_effect(key: Array, expr: Array, mu: float, sigma: float) -> Array:
    """Rescale each cell to a log-normal target library size."""
    n_cells = expr.shape[0]
    lib = _lognormal(key, (n_cells,), mu, sigma)  # (cells,)
    norm = jnp.sum(expr, axis=1)  # (cells,) current library size per cell
    cell_scale = lib / jnp.where(norm > 0, norm, 1.0)
    return expr * cell_scale[:, None]


def add_dropout(key: Array, expr: Array, k: float, q: float) -> Array:
    """Zero out low-expression entries via a logistic dropout on ``log(x+1)``."""
    log_expr = jnp.log1p(expr)
    mid = jnp.percentile(log_expr, q, method="linear")
    keep_p = 1.0 / (1.0 + jnp.exp(-k * (log_expr - mid)))  # P(keep)
    drop = random.bernoulli(key, 1.0 - keep_p, expr.shape)
    return jnp.where(drop, 0.0, expr)


def to_umi_counts(key: Array, expr: Array) -> Array:
    """Sample integer UMI counts as ``Poisson(expr)`` (0 where ``expr <= 0``)."""
    lam = jnp.maximum(expr, 0.0)
    counts = random.poisson(key, lam, expr.shape).astype(expr.dtype)
    return jnp.where(expr > 0, counts, 0.0)


def add_technical_noise(key: Array, expr: Array, profile: NoiseProfile) -> Array:
    """Full technical-noise pipeline on a ``(num_cells, num_genes)`` matrix."""
    k1, k2, k3, k4 = random.split(key, 4)
    expr = add_outlier_effect(k1, expr, profile.outlier_p, profile.outlier_mu, profile.outlier_sigma)
    expr = add_library_size_effect(k2, expr, profile.library_mu, profile.library_sigma)
    expr = add_dropout(k3, expr, profile.dropout_k, profile.dropout_q)
    expr = to_umi_counts(k4, expr)
    return expr


def add_technical_noise_by_name(key: Array, expr: Array, profile: str = "DS6") -> Array:
    """Convenience wrapper selecting a DS profile by name."""
    return add_technical_noise(key, expr, DS_PROFILES[profile])
