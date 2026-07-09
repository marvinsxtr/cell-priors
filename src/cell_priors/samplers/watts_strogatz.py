"""Watts–Strogatz small-world GRN sampler, as a single jitted function."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ..base import GRN, GRNSampler
from .grouped_scale_free import edges_to_grn


def watts_strogatz_edges(
    key: Array,
    n: int,
    k_neighbors: int = 2,
    beta: float = 0.1,
) -> tuple[Array, Array, Array, Array]:
    """Sample a directed Watts–Strogatz small-world graph as a fixed-size masked edge list.

    Starts from a ring lattice in which every node points to its ``k_neighbors`` nearest
    clockwise neighbours, then rewires each edge's target to a uniformly random node with
    probability ``beta``. Small ``beta`` preserves the lattice's high local clustering
    (short-range structure); ``beta -> 1`` approaches a random graph.

    Args:
        key: PRNG key.
        n: Number of nodes (static).
        k_neighbors: Out-degree of the ring lattice (static; sets the edge count ``n * k_neighbors``).
        beta: Per-edge rewiring probability (may be traced).

    Returns:
        ``(sources, targets, edge_valid, groups)``: ``(n * k_neighbors,)`` int endpoint
        arrays, a float validity mask (0 for a self-loop produced by rewiring), and an
        all-zero ``(n,)`` group label (the model is unstructured).
    """
    if n < 2:
        raise ValueError("num_genes must be >= 2.")
    k_neighbors = int(k_neighbors)
    if not 1 <= k_neighbors < n:
        raise ValueError("k_neighbors must satisfy 1 <= k_neighbors < num_genes.")

    e = n * k_neighbors
    sources = jnp.repeat(jnp.arange(n), k_neighbors)
    offsets = jnp.tile(jnp.arange(1, k_neighbors + 1), n)
    lattice_targets = (sources + offsets) % n

    k_flip, k_rand = jax.random.split(key)
    rewire = jax.random.uniform(k_flip, (e,)) < beta
    random_targets = jax.random.randint(k_rand, (e,), 0, n)
    targets = jnp.where(rewire, random_targets, lattice_targets)

    valid = (sources != targets).astype(jnp.float32)
    groups = jnp.zeros(n, jnp.int32)
    return sources.astype(jnp.int32), targets.astype(jnp.int32), valid, groups


class WattsStrogatzSampler(GRNSampler):
    """Directed Watts–Strogatz small-world GRN sampler."""

    def __init__(self, k_neighbors: int = 2, beta: float = 0.1) -> None:
        self.k_neighbors = k_neighbors
        self.beta = beta

    def sample(self, key: Array, num_genes: int, **kwargs: object) -> GRN:
        k_neighbors = kwargs.get("k_neighbors", self.k_neighbors)
        beta = kwargs.get("beta", self.beta)
        sources, targets, valid, groups = watts_strogatz_edges(key, num_genes, k_neighbors, beta)
        return edges_to_grn(sources, targets, valid, groups)
