"""Pure-JAX core of the grn-paper expression model (Aguirre et al. 2025).

Reimplements ``grn.simulate_rna``: a sigmoid-link stochastic differential equation
driven by a signed interaction matrix (``beta[i, j]`` = effect of regulator ``i`` on
target ``j``), per-gene basal log-production ``alpha`` and degradation ``l``::

    X(t+dt) = X(t) + dt * (sigmoid(alpha + Xbeta) - l*X)  +  s * sqrt(dt*X) * N(0,1)

clipped at 0. Each "cell" is an independent SDE realization; its observed
expression is the time-average over the post-burn-in window (the reference's
observation model). The time integration is a single ``lax.scan`` that accumulates
the running mean instead of storing the trajectory, so memory is ``O(G)`` per cell.

The reference stores ``beta`` densely, but it is ``S * E`` -- elementwise product of
a Gaussian matrix with the (sparse) edge-multiplicity matrix -- so it is nonzero
*only on actual edges*. We therefore store ``beta`` as a sparse edge list and compute
``X @ beta`` with a single ``segment_sum``, making each step ``O(cells * E)`` instead
of ``O(cells * G^2)``. This is numerically identical to the dense product.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax, random


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GrnPaperParams:
    """Parameters of the grn-paper expression model (a pytree of arrays).

    The interaction matrix is stored sparsely as directed edges ``reg -> tar`` with
    a signed per-edge weight ``beta`` (the value the dense matrix would hold there).
    """

    reg_idx: Array  # (E,) int: regulator gene index
    tar_idx: Array  # (E,) int: target gene index
    beta: Array  # (E,) float: signed interaction weight of edge reg -> tar
    alpha: Array  # (G,) basal log-production (pre-sigmoid)
    l: Array  # (G,) degradation rate
    group: Array  # (G,) module label (carried through from the GRN)

    @property
    def num_genes(self) -> int:
        return self.alpha.shape[0]

    @property
    def num_edges(self) -> int:
        return self.reg_idx.shape[0]

    @classmethod
    def from_dense(cls, beta, alpha, l, group=None) -> GrnPaperParams:
        """Build params from a dense ``(G, G)`` interaction matrix ``beta``.

        Off-diagonal nonzeros of ``beta`` become edges; the result simulates
        identically to the dense matrix.
        """
        beta = np.asarray(beta)
        reg, tar = np.nonzero(beta)
        alpha = jnp.asarray(alpha)
        group = jnp.zeros(alpha.shape[0], jnp.int32) if group is None else jnp.asarray(group)
        return cls(
            reg_idx=jnp.asarray(reg, dtype=jnp.int32),
            tar_idx=jnp.asarray(tar, dtype=jnp.int32),
            beta=jnp.asarray(beta[reg, tar]),
            alpha=alpha,
            l=jnp.asarray(l),
            group=group,
        )


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


def _regulatory_input(params: GrnPaperParams, x: Array) -> Array:
    """Compute ``X @ beta`` sparsely: shape ``(cells, G)``.

    For each edge ``i -> j`` accumulates ``beta_ij * x_i`` into target ``j`` with a
    single batched scatter-add (faster than a transpose + ``segment_sum``).
    """
    contrib = params.beta[None, :] * x[:, params.reg_idx]  # (cells, E)
    out = jnp.zeros_like(x)
    return out.at[:, params.tar_idx].add(contrib)


def _step(params: GrnPaperParams, x: Array, key: Array, dt: float, s: float) -> Array:
    """One Euler-Maruyama step for a batch of cells ``x`` of shape ``(cells, G)``."""
    prod = jax.nn.sigmoid(params.alpha + _regulatory_input(params, x))  # (cells, G)
    drift = prod - params.l * x
    noise = s * jnp.sqrt(dt * x) * random.normal(key, x.shape)
    return jnp.maximum(0.0, x + dt * drift + noise)


def simulate(params: GrnPaperParams, key: Array, cfg: GrnPaperConfig) -> Array:
    """Integrate the SDE and return time-averaged expression, shape ``(cells, G)``.

    Two ``lax.scan`` phases: a burn-in that only advances the state, then a
    sampling phase that accumulates the running mean (avoids a per-step
    branch/accumulator during burn-in).
    """
    g = params.num_genes
    k_burn, k_sample = random.split(key)

    def advance(x, key_t):
        return _step(params, x, key_t, cfg.dt, cfg.s), None

    def accumulate(carry, key_t):
        x, acc = carry
        x = _step(params, x, key_t, cfg.dt, cfg.s)
        return (x, acc + x), None

    x0 = jnp.zeros((cfg.num_cells, g), dtype=params.alpha.dtype)
    x_burned, _ = lax.scan(advance, x0, random.split(k_burn, cfg.burnin))
    (_, acc), _ = lax.scan(accumulate, (x_burned, jnp.zeros_like(x0)), random.split(k_sample, cfg.sample_steps))
    return acc / cfg.sample_steps


def _edge_scale(params: GrnPaperParams, gene_indices: Array, value: float) -> Array:
    """Per-edge multiplier that sets edges leaving ``gene_indices`` to ``value``."""
    gene_scale = jnp.ones(params.num_genes, dtype=params.beta.dtype)
    gene_scale = gene_scale.at[jnp.asarray(gene_indices).reshape(-1)].set(value)
    return gene_scale[params.reg_idx]


def knockout(params: GrnPaperParams, gene_indices: Array) -> GrnPaperParams:
    """Hard knockout: zero the genes' outgoing interactions (reference KO)."""
    return dataclasses.replace(params, beta=params.beta * _edge_scale(params, gene_indices, 0.0))


def knockdown(params: GrnPaperParams, gene_indices: Array, strength: float = 1.0) -> GrnPaperParams:
    """Soft knockdown: attenuate the genes' outgoing interactions by ``strength``."""
    return dataclasses.replace(params, beta=params.beta * _edge_scale(params, gene_indices, 1.0 - strength))
