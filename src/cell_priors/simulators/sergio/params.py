"""GRN representation for the JAX SERGIO prior.

The GRN is stored as a *sparse edge list* (regulator/target index arrays) rather
than a dense adjacency matrix so the hot simulation loop scales with the number
of edges ``E`` instead of ``num_genes**2``. All per-edge and per-gene quantities
are plain JAX arrays, so :class:`SergioParams` is a pytree and the whole prior
can live inside a single jitted/vmapped computation graph alongside a model.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SergioParams:
    """Parameters of a SERGIO gene-regulatory network (a pytree of arrays).

    Edges are directed ``reg -> tar``. ``edge_mask``, ``prod_scale`` and
    ``ko_mask`` are intervention knobs (all neutral by default) so that an
    intervened network is just another ``SergioParams`` with the same shapes.
    """

    # Per-edge quantities, shape (E,)
    reg_idx: Array  # int: regulator gene index
    tar_idx: Array  # int: target gene index
    k: Array  # float: signed interaction strength (>0 activation, <0 repression)
    hill_n: Array  # float: Hill coefficient
    h: Array  # float: half-response (filled by `init`; zeros before)
    edge_mask: Array  # float {0,1}: 0 removes the edge (hard knockout)

    # Per-gene quantities, shape (G,)
    decay: Array  # float: mRNA decay rate (lambda)
    mr_mask: Array  # float {0,1}: 1 if gene is a master regulator (no active in-edges)
    prod_scale: Array  # float: production multiplier (soft CRISPRi knockdown)
    ko_mask: Array  # float {0,1}: 1 forces the gene's expression to zero

    # Per-gene-per-cell-type, shape (G, C)
    prod_rates: Array  # float: master-regulator basal production per cell type

    @property
    def num_genes(self) -> int:
        return self.decay.shape[0]

    @property
    def num_edges(self) -> int:
        return self.reg_idx.shape[0]

    @property
    def num_cell_types(self) -> int:
        return self.prod_rates.shape[1]


@dataclass(frozen=True)
class SergioConfig:
    """Static simulation hyperparameters (passed as a jit static argument)."""

    num_cells: int = 200
    num_cell_types: int = 1
    safety_iter: int = 150
    scale_iter: int = 10
    dt: float = 0.01
    noise_s: float = 1.0
    init_iters: int | None = None  # steady-state fixed-point iterations; None -> num_genes
    require_mrs: bool = True  # False -> basal production for every gene (cycle-tolerant)

    @property
    def max_iter(self) -> int:
        return self.num_cells * self.scale_iter + self.safety_iter


def recompute_mr_mask(p: SergioParams) -> SergioParams:
    """Recompute which genes are master regulators from the active edges.

    A gene with no active incoming edges is a master regulator and is driven by
    ``prod_rates``. This mirrors SERGIO's ``set_mrs`` and, crucially, makes genes
    that become orphaned by a knockout (their sole regulator removed) turn into
    master regulators automatically.
    """
    g = p.num_genes
    in_active = jax.ops.segment_sum(p.edge_mask, p.tar_idx, num_segments=g)
    mr_mask = (in_active == 0).astype(p.decay.dtype)
    return dataclasses.replace(p, mr_mask=mr_mask)


def make_params(
    reg_idx: np.ndarray,
    tar_idx: np.ndarray,
    k: np.ndarray,
    hill_n: np.ndarray,
    decay: np.ndarray,
    prod_rates: np.ndarray,
    dtype=jnp.float32,
) -> SergioParams:
    """Build neutral (un-intervened) :class:`SergioParams` from raw arrays.

    ``prod_rates`` has shape ``(num_genes, num_cell_types)`` and is sampled for
    *every* gene (not only master regulators) so orphaned genes have a basal rate.
    """
    e = len(reg_idx)
    g = len(decay)
    p = SergioParams(
        reg_idx=jnp.asarray(reg_idx, dtype=jnp.int32),
        tar_idx=jnp.asarray(tar_idx, dtype=jnp.int32),
        k=jnp.asarray(k, dtype=dtype),
        hill_n=jnp.asarray(hill_n, dtype=dtype),
        h=jnp.zeros(e, dtype=dtype),
        edge_mask=jnp.ones(e, dtype=dtype),
        decay=jnp.asarray(decay, dtype=dtype),
        mr_mask=jnp.zeros(g, dtype=dtype),
        prod_scale=jnp.ones(g, dtype=dtype),
        ko_mask=jnp.zeros(g, dtype=dtype),
        prod_rates=jnp.asarray(prod_rates, dtype=dtype),
    )
    return recompute_mr_mask(p)


def build_sergio_params(
    grn,
    key: Array,
    num_cell_types: int = 1,
    decay_range: tuple[float, float] = (0.5, 1.0),
    hill_n_range: tuple[float, float] = (1.5, 2.5),
    interaction_k_range: tuple[float, float] = (1.0, 5.0),
    repression_prob_range: tuple[float, float] = (0.0, 0.5),
    mr_low_range: tuple[float, float] = (0.5, 2.0),
    mr_high_range: tuple[float, float] = (3.0, 5.0),
    dtype=jnp.float32,
) -> SergioParams:
    """Sample SERGIO kinetics for a :class:`GRN` as a pure JAX function.

    Keeps the graph exactly as drawn (no cycle removal) and gives every gene its own
    basal production rate, so any structure -- cyclic, source-free -- is driven. A unique
    edge is active where the GRN's ``weight`` is positive (one row per ``reg -> tar`` pair,
    each with its own sampled interaction); duplicate / padding / self-loop slots are
    masked out. (SERGIO has no multiplicity notion, so the count itself is unused here.)

    Because it only uses ``jax.random`` and fixed shapes, this composes with the
    sampler and simulator inside a single ``jit``/``vmap``-ed graph.
    """
    e = grn.num_edges
    g = grn.num_genes
    c = num_cell_types
    k_hill, k_mag, k_sign, k_rep, k_decay, k_hi, k_lo, k_mix = jax.random.split(key, 8)

    def _u(k, shape, rng):
        return jax.random.uniform(k, shape, minval=rng[0], maxval=rng[1])

    hill_n = _u(k_hill, (e,), hill_n_range)
    k_abs = _u(k_mag, (e,), interaction_k_range)
    # One repression probability per network (drawn from the range), then each edge is
    # repressing (signed -1) with that probability -- mirrors how decay/hill/k are ranged.
    repression_prob = _u(k_rep, (), repression_prob_range)
    sign = jnp.where(jax.random.uniform(k_sign, (e,)) < repression_prob, -1.0, 1.0)
    decay = _u(k_decay, (g,), decay_range)
    high = _u(k_hi, (g, c), mr_high_range)
    low = _u(k_lo, (g, c), mr_low_range)
    prod_rates = jnp.where(jax.random.uniform(k_mix, (g, c)) < 0.5, high, low)

    p = SergioParams(
        reg_idx=grn.reg_idx.astype(jnp.int32),
        tar_idx=grn.tar_idx.astype(jnp.int32),
        k=(k_abs * sign).astype(dtype),
        hill_n=hill_n.astype(dtype),
        h=jnp.zeros(e, dtype=dtype),
        edge_mask=(grn.weight > 0).astype(dtype),
        decay=decay.astype(dtype),
        mr_mask=jnp.zeros(g, dtype=dtype),
        prod_scale=jnp.ones(g, dtype=dtype),
        ko_mask=jnp.zeros(g, dtype=dtype),
        prod_rates=prod_rates.astype(dtype),
    )
    return recompute_mr_mask(p)
