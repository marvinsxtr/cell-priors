"""Jittable greedy cycle removal for turning a sampled GRN into a DAG.

Canonical greedy: repeatedly drop the globally minimum-``abs_weight`` edge that still
lies on a cycle, until the kept subgraph is acyclic. Order-independent (with distinct
weights the result is unique), so it composes inside ``jit``/``vmap`` -- unlike the host
networkx variant that breaks one arbitrarily-found cycle at a time.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array


def _reachable(adj: Array) -> Array:
    """Length->=1 reachability of a boolean adjacency via repeated squaring.

    Args:
        adj: gene-by-gene adjacency, ``(G, G)`` with 1 where a kept edge ``u->v`` exists.

    Returns:
        ``(G, G)`` mask where entry ``(a, b)`` is 1 iff a directed path ``a->b`` of
        length >= 1 exists over the kept edges.
    """
    g = adj.shape[-1]
    steps = max(1, math.ceil(math.log2(g))) if g > 1 else 1
    reach = adj
    for _ in range(steps):
        reach = ((reach + reach @ reach) > 0).astype(adj.dtype)
    return reach


def dag_edge_mask(reg_idx: Array, tar_idx: Array, abs_weight: Array, active: Array, num_genes: int) -> Array:
    """Keep-mask selecting an acyclic subset of the active edges.

    Args:
        reg_idx: regulator gene index per edge, ``(E,)``.
        tar_idx: target gene index per edge, ``(E,)``.
        abs_weight: non-negative edge weight (interaction magnitude), ``(E,)``.
        active: initial per-edge active flag, ``(E,)`` (padding/self-loop slots 0).
        num_genes: number of genes (static).

    Returns:
        Boolean ``(E,)`` mask, a subset of ``active`` whose kept edges form a DAG.
    """
    g = num_genes
    e = reg_idx.shape[-1]
    keep0 = active > 0

    def body(_: Array, keep: Array) -> Array:
        adj = jnp.zeros((g, g), jnp.float32).at[reg_idx, tar_idx].max(keep.astype(jnp.float32))
        reach = _reachable(adj)
        on_cycle = keep & (reach[tar_idx, reg_idx] > 0)
        masked_w = jnp.where(on_cycle, abs_weight, jnp.inf)
        worst = jnp.argmin(masked_w)
        drop = jnp.any(on_cycle)
        return keep.at[worst].set(jnp.where(drop, False, keep[worst]))

    return jax.lax.fori_loop(0, e, body, keep0)
