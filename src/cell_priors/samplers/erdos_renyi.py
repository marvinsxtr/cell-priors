"""Erdős–Rényi random-graph GRN sampler, as a single jitted function."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ..base import GRN, GRNSampler
from .grouped_scale_free import edges_to_grn


def erdos_renyi_edges(
    key: Array,
    n: int,
    mean_degree: float = 2.0,
    max_edges: int | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Sample a directed Erdős–Rényi random graph as a fixed-size masked edge list.

    Each of ``max_edges`` candidate edges draws both endpoints uniformly at random; the
    first ``round(mean_degree * n)`` non-self-loop candidates are kept. The result is a
    homogeneous graph with Poisson-like degrees and no hubs or modular structure -- the
    standard null model against which structured samplers are compared.

    Args:
        key: PRNG key.
        n: Number of nodes (static; sets the group-label shape).
        mean_degree: Expected number of edges per node (may be traced); the kept edge
            count is ``round(mean_degree * n)``, capped at ``max_edges``.
        max_edges: Fixed edge-buffer size (static); defaults to ``4 * n``.

    Returns:
        ``(sources, targets, edge_valid, groups)``: ``(max_edges,)`` int endpoint arrays,
        a ``(max_edges,)`` float validity mask (1 for a kept non-self-loop edge, else 0),
        and an all-zero ``(n,)`` group label (the model is unstructured).
    """
    if n < 2:
        raise ValueError("num_genes must be >= 2.")
    if max_edges is None:
        max_edges = 4 * n
    max_edges = int(max_edges)

    k_src, k_tar = jax.random.split(key)
    sources = jax.random.randint(k_src, (max_edges,), 0, n)
    targets = jax.random.randint(k_tar, (max_edges,), 0, n)

    num_edges = jnp.round(jnp.asarray(mean_degree) * n).astype(jnp.int32)
    valid = (jnp.arange(max_edges) < num_edges) & (sources != targets)
    groups = jnp.zeros(n, jnp.int32)
    return sources.astype(jnp.int32), targets.astype(jnp.int32), valid.astype(jnp.float32), groups


class ErdosRenyiSampler(GRNSampler):
    """Directed Erdős–Rényi random-graph GRN sampler (unstructured null model)."""

    def __init__(self, mean_degree: float = 2.0) -> None:
        self.mean_degree = mean_degree

    def sample(self, key: Array, num_genes: int, max_edges: int | None = None, **kwargs: object) -> GRN:
        mean_degree = kwargs.get("mean_degree", self.mean_degree)
        sources, targets, valid, groups = erdos_renyi_edges(key, num_genes, mean_degree, max_edges)
        return edges_to_grn(sources, targets, valid, groups)
