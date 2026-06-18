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


def random_dag_edges(
    rng: np.random.Generator,
    num_genes: int,
    avg_regulators: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a random DAG edge set (regulator index < target index).

    Drawing edges only from lower- to higher-indexed genes guarantees acyclicity
    and that the lowest-indexed genes act as master regulators. ``avg_regulators``
    controls the expected in-degree.
    """
    regs: list[int] = []
    tars: list[int] = []
    for tar in range(1, num_genes):
        # expected number of regulators, capped by how many lower-index genes exist
        n_reg = min(tar, rng.poisson(avg_regulators))
        n_reg = max(n_reg, 1) if tar == 1 else n_reg
        if n_reg == 0:
            continue
        chosen = rng.choice(tar, size=n_reg, replace=False)
        for reg in chosen:
            regs.append(int(reg))
            tars.append(tar)
    return np.asarray(regs, dtype=np.int64), np.asarray(tars, dtype=np.int64)


def sample_random_params(
    key_or_seed,
    num_genes: int,
    num_cell_types: int = 1,
    avg_regulators: float = 2.0,
    decay_range: tuple[float, float] = (0.5, 1.0),
    hill_n_range: tuple[float, float] = (1.5, 2.5),
    interaction_k_range: tuple[float, float] = (1.0, 5.0),
    repression_prob: float = 0.0,
    mr_low_range: tuple[float, float] = (0.5, 2.0),
    mr_high_range: tuple[float, float] = (3.0, 5.0),
    dtype=jnp.float32,
) -> SergioParams:
    """Sample a random SERGIO GRN (structure + kinetic parameters).

    Production rates are drawn per gene per cell type as a Bernoulli mixture of a
    low and a high uniform range, matching SERGIO's ``MrProfile.from_random``.
    """
    seed = int(key_or_seed) if np.isscalar(key_or_seed) else int(jax.random.randint(key_or_seed, (), 0, 2**31))
    rng = np.random.default_rng(seed)

    reg_idx, tar_idx = random_dag_edges(rng, num_genes, avg_regulators)
    e = len(reg_idx)
    decay = rng.uniform(*decay_range, size=num_genes)
    hill_n = rng.uniform(*hill_n_range, size=e)
    k_mag = rng.uniform(*interaction_k_range, size=e)
    sign = np.where(rng.random(e) < repression_prob, -1.0, 1.0)
    k = k_mag * sign

    # Bernoulli(0.5) mixture of low/high uniform ranges, per gene per cell type.
    high = rng.random((num_genes, num_cell_types)) < 0.5
    low_vals = rng.uniform(*mr_low_range, size=(num_genes, num_cell_types))
    high_vals = rng.uniform(*mr_high_range, size=(num_genes, num_cell_types))
    prod_rates = np.where(high, high_vals, low_vals)

    return make_params(reg_idx, tar_idx, k, hill_n, decay, prod_rates, dtype=dtype)
