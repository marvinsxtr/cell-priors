"""Pure-JAX core of the grn-paper expression model (Aguirre et al. 2025).

Reimplements ``grn.simulate_rna``: a sigmoid-link stochastic differential equation
on a dense signed interaction matrix ``beta`` (``beta[i, j]`` = effect of regulator
``i`` on target ``j``), per-gene basal log-production ``alpha`` and degradation
``l``::

    X(t+dt) = X(t) + dt * (sigmoid(alpha + Xbeta) - l*X)  +  s * sqrt(dt*X) * N(0,1)

clipped at 0. Each "cell" is an independent SDE realization; its observed
expression is the time-average over the post-burn-in window (the reference's
observation model). The time integration is a single ``lax.scan`` that accumulates
the running mean instead of storing the trajectory, so memory is ``O(G)`` per cell.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array, lax, random


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GrnPaperParams:
    """Parameters of the grn-paper expression model (a pytree of arrays)."""

    beta: Array  # (G, G) signed interaction matrix; beta[i, j] = effect of i on j
    alpha: Array  # (G,) basal log-production (pre-sigmoid)
    l: Array  # (G,) degradation rate
    group: Array  # (G,) module label (carried through from the GRN)

    @property
    def num_genes(self) -> int:
        return self.alpha.shape[0]

    @classmethod
    def from_dense(cls, beta, alpha, l, group=None) -> "GrnPaperParams":
        """Build params from a dense ``(G, G)`` interaction matrix ``beta``."""
        alpha = jnp.asarray(alpha)
        group = jnp.zeros(alpha.shape[0], jnp.int32) if group is None else jnp.asarray(group)
        return cls(beta=jnp.asarray(beta), alpha=alpha, l=jnp.asarray(l), group=group)


@dataclass(frozen=True)
class GrnPaperConfig:
    """Static integration hyperparameters."""

    num_cells: int = 64
    n_steps: int = 4000
    burnin: int = 2000
    dt: float = 1e-2
    s: float = 1e-4

    @property
    def sample_steps(self) -> int:
        return self.n_steps - self.burnin


def _step(params: GrnPaperParams, x: Array, key: Array, dt: float, s: float) -> Array:
    """One Euler-Maruyama step for a batch of cells ``x`` of shape ``(cells, G)``."""
    prod = jax.nn.sigmoid(params.alpha + x @ params.beta)  # (cells, G)
    drift = prod - params.l * x
    noise = s * jnp.sqrt(dt * x) * random.normal(key, x.shape)
    return jnp.maximum(0.0, x + dt * drift + noise)


def simulate(params: GrnPaperParams, key: Array, cfg: GrnPaperConfig) -> Array:
    """Integrate the SDE and return time-averaged expression, shape ``(cells, G)``."""
    g = params.num_genes
    x0 = jnp.zeros((cfg.num_cells, g), dtype=params.alpha.dtype)

    def body(carry, key_t):
        x, acc, i = carry
        x = _step(params, x, key_t, cfg.dt, cfg.s)
        acc = acc + jnp.where(i >= cfg.burnin, x, 0.0)  # accumulate only post-burn-in
        return (x, acc, i + 1), None

    keys = random.split(key, cfg.n_steps)
    (_, acc, _), _ = lax.scan(body, (x0, jnp.zeros_like(x0), 0), keys)
    return acc / cfg.sample_steps


def knockout(params: GrnPaperParams, gene_indices: Array) -> GrnPaperParams:
    """Hard knockout: zero the genes' outgoing interactions (reference KO)."""
    keep = jnp.ones(params.num_genes, dtype=params.beta.dtype).at[jnp.asarray(gene_indices).reshape(-1)].set(0.0)
    return dataclasses.replace(params, beta=params.beta * keep[:, None])


def knockdown(params: GrnPaperParams, gene_indices: Array, strength: float = 1.0) -> GrnPaperParams:
    """Soft knockdown: attenuate the genes' outgoing interactions by ``strength``."""
    scale = jnp.ones(params.num_genes, dtype=params.beta.dtype)
    scale = scale.at[jnp.asarray(gene_indices).reshape(-1)].set(1.0 - strength)
    return dataclasses.replace(params, beta=params.beta * scale[:, None])
