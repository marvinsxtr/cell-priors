"""Barabási–Albert scale-free GRN sampler, as a single jitted scan."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array, lax

from ..base import GRN, GRNSampler
from .grouped_scale_free import edges_to_grn


def scale_free_edges(
    key: Array,
    n: int,
    m: int = 2,
    delta: float = 1.0,
) -> tuple[Array, Array, Array, Array]:
    """Grow a directed Barabási–Albert scale-free graph as a jitted scan.

    Nodes are added one at a time; each new node receives ``m`` incoming edges from
    existing nodes chosen by preferential attachment on out-degree (probability
    ``out_degree + delta``). A few nodes therefore accumulate very high out-degree (hub
    regulators) and the out-degree distribution is heavy-tailed, with a single module and
    no small-world clustering -- the pure preferential-attachment counterpart to the
    grouped scale-free sampler.

    Args:
        key: PRNG key.
        n: Number of nodes (static).
        m: Edges added per new node (static; sets the edge count ``(n - m) * m``).
        delta: Attachment smoothing added to each node's out-degree (may be traced);
            larger values flatten the degree distribution towards uniform attachment.

    Returns:
        ``(sources, targets, edge_valid, groups)``: ``((n - m) * m,)`` int endpoint
        arrays, an all-ones float validity mask, and an all-zero ``(n,)`` group label.
    """
    m = int(m)
    if not 1 <= m < n:
        raise ValueError("m must satisfy 1 <= m < num_genes.")

    node_idx = jnp.arange(n)
    out_deg0 = jnp.zeros(n, jnp.float32)

    def step(out_deg: Array, cur_and_key: tuple[Array, Array]):
        cur, step_key = cur_and_key
        valid = node_idx < cur
        logits = jnp.where(valid, jnp.log(out_deg + delta), -jnp.inf)
        pick_keys = jax.random.split(step_key, m)
        srcs = jax.vmap(lambda pk: jax.random.categorical(pk, logits))(pick_keys)
        out_deg = out_deg.at[srcs].add(1.0)
        return out_deg, (srcs.astype(jnp.int32), jnp.full((m,), cur, jnp.int32))

    new_nodes = jnp.arange(m, n)
    step_keys = jax.random.split(key, n - m)
    _, (srcs, tars) = lax.scan(step, out_deg0, (new_nodes, step_keys))

    sources = srcs.reshape(-1)
    targets = tars.reshape(-1)
    valid = jnp.ones_like(sources, jnp.float32)
    groups = jnp.zeros(n, jnp.int32)
    return sources.astype(jnp.int32), targets.astype(jnp.int32), valid, groups


class ScaleFreeSampler(GRNSampler):
    """Directed Barabási–Albert scale-free GRN sampler (single-module preferential attachment)."""

    def __init__(self, m: int = 2, delta: float = 1.0) -> None:
        self.m = m
        self.delta = delta

    def sample(self, key: Array, num_genes: int, **kwargs: object) -> GRN:
        m = kwargs.get("m", self.m)
        delta = kwargs.get("delta", self.delta)
        sources, targets, valid, groups = scale_free_edges(key, num_genes, m, delta)
        return edges_to_grn(sources, targets, valid, groups)
