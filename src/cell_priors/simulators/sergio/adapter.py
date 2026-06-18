"""Adapt a sampled :class:`GRN` structure into SERGIO kinetic parameters.

SERGIO requires a *directed acyclic* regulatory graph (it estimates a steady state
by a topological pass), whereas general samplers -- e.g. the grouped scale-free
sampler -- produce cyclic graphs. This module assigns signed interaction
strengths, Hill coefficients, decay rates and master-regulator production rates,
and breaks any cycles greedily (removing the weakest edge in each cycle, as in
MapPFN's SERGIO dataset), yielding a :class:`SergioParams` the validated core can
simulate.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from jax import Array

from ...base import GRN
from .params import SergioParams, make_params


def _acyclic_edge_mask(reg: np.ndarray, tar: np.ndarray, abs_weight: np.ndarray, num_genes: int) -> np.ndarray:
    """Boolean mask of edges to keep so the graph is a DAG.

    Repeatedly finds a cycle and drops its smallest-``abs_weight`` edge.
    """
    g = nx.DiGraph()
    g.add_nodes_from(range(num_genes))
    for e, (r, t) in enumerate(zip(reg.tolist(), tar.tolist())):
        g.add_edge(r, t, idx=e, w=float(abs_weight[e]))
    keep = np.ones(len(reg), dtype=bool)
    while not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g)
        u, v = min(cycle, key=lambda uv: g[uv[0]][uv[1]]["w"])
        keep[g[u][v]["idx"]] = False
        g.remove_edge(u, v)
    return keep


def grn_to_sergio_params(
    grn: GRN,
    key: Array,
    num_cell_types: int = 1,
    decay_range: tuple[float, float] = (0.5, 1.0),
    hill_n_range: tuple[float, float] = (1.5, 2.5),
    interaction_k_range: tuple[float, float] = (1.0, 5.0),
    repression_prob: float = 0.0,
    mr_low_range: tuple[float, float] = (0.5, 2.0),
    mr_high_range: tuple[float, float] = (3.0, 5.0),
    acyclic: bool = True,
    dtype=None,
) -> SergioParams:
    """Build :class:`SergioParams` from a :class:`GRN` (host-side, seeded by ``key``).

    ``acyclic=True`` (default) breaks cycles so standard SERGIO's topological
    steady-state estimate applies. Set ``acyclic=False`` to keep the graph as-is
    (used by the cycle-tolerant MapPFN prior, which also drops the master-regulator
    requirement via ``SergioConfig.require_mrs=False``).
    """
    import jax

    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.default_rng(seed)

    reg = np.asarray(grn.reg_idx)
    tar = np.asarray(grn.tar_idx)
    num_genes = grn.num_genes
    e = len(reg)

    hill_n = rng.uniform(*hill_n_range, size=e)
    k_mag = rng.uniform(*interaction_k_range, size=e)
    sign = np.where(rng.random(e) < repression_prob, -1.0, 1.0)
    k = k_mag * sign

    if acyclic:
        # Break cycles using |k| as the edge weight, then keep the acyclic subset.
        keep = _acyclic_edge_mask(reg, tar, np.abs(k), num_genes)
        reg, tar, k, hill_n = reg[keep], tar[keep], k[keep], hill_n[keep]

    decay = rng.uniform(*decay_range, size=num_genes)
    high = rng.random((num_genes, num_cell_types)) < 0.5
    prod_rates = np.where(
        high,
        rng.uniform(*mr_high_range, size=(num_genes, num_cell_types)),
        rng.uniform(*mr_low_range, size=(num_genes, num_cell_types)),
    )

    import jax.numpy as jnp

    return make_params(reg, tar, k, hill_n, decay, prod_rates, dtype=dtype or jnp.float32)
