"""Grouped scale-free GRN sampler (Aguirre et al. 2025), reimplemented in JAX.

A JAX port of ``grouped_scale_free_graph`` from the grn-paper. The graph grows by
Bollobas-style directed preferential attachment with three moves (probabilities
``alpha``/``beta``/``gamma`` summing to 1) and an added group structure: ``k``
modules with within-group preferential attachment controlled by ``kappa``.

Preferential attachment is inherently sequential -- each edge's endpoint
distribution depends on the degrees accumulated so far -- so structure growth is a
host-driven Python loop. All randomness goes through ``jax.random`` (a per-step
split of the user's key, ``uniform`` for the move and ``categorical`` for endpoint
selection), and the result is returned as JAX arrays, ready for a jittable
simulator. The numerics that matter for training speed live in the simulators,
which are fully ``jit``/``vmap``-able.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..base import GRN, GRNSampler


def grouped_scale_free_edges(
    key: Array,
    n: int,
    alpha: float = 0.05,
    beta: float = 0.54,
    gamma: float = 0.41,
    delta_in: float = 2.0,
    delta_out: float = 0.0,
    k: int = 1,
    kappa: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Grow a grouped scale-free directed multigraph; return ``(sources, targets, groups)``.

    Mirrors the reference algorithm exactly: 3-cycle seed, group seeding by ``k``,
    and the alpha/beta/gamma moves. ``categorical`` draws replace the reference's
    ``np.random.choice(..., p=...)`` (logits = log of the same attachment scores).
    """
    if abs(alpha + beta + gamma - 1.0) >= 1e-9:
        raise ValueError("alpha + beta + gamma must equal 1.")
    for name, val in (
        ("alpha", alpha),
        ("beta", beta),
        ("gamma", gamma),
        ("delta_in", delta_in),
        ("delta_out", delta_out),
        ("kappa", kappa),
    ):
        if val < 0:
            raise ValueError(f"{name} must be >= 0.")
    k = max(int(k), 1)

    # Seed with a 3-cycle. Group assignment of the seed matches the reference.
    sources = [0, 1, 2]
    targets = [1, 2, 0]
    if k < 3:
        groups = {0: 0, 1: 1, 2: 0} if k == 2 else {0: 0, 1: 0, 2: 0}
    else:
        groups = {0: 0, 1: 1, 2: 2}
    group_members: dict[int, set[int]] = {}
    for node, g in groups.items():
        group_members.setdefault(g, set()).add(node)

    d_in = [targets.count(i) for i in range(3)]
    d_out = [sources.count(i) for i in range(3)]

    key_seq = iter(_key_stream(key))

    def _categorical(scores: np.ndarray) -> int:
        logits = jnp.log(jnp.asarray(scores))
        return int(jax.random.categorical(next(key_seq), logits))

    def _group_bonus(target_group: int, n_nodes: int) -> np.ndarray:
        bonus = np.ones(n_nodes)
        for member in group_members[target_group]:
            bonus[member] = kappa
        return bonus

    while len(d_in) < n:
        n_nodes = len(d_in)
        r = float(jax.random.uniform(next(key_seq)))
        if r < gamma:
            # New target node v; pick existing source w by out-degree, biased to v's group.
            v = n_nodes
            d_in.append(0)
            d_out.append(0)
            g_v = int(jax.random.randint(next(key_seq), (), 0, k))
            group_members.setdefault(g_v, set()).add(v)
            groups[v] = g_v
            scores = _group_bonus(g_v, len(d_out)) * (np.asarray(d_out) + delta_out)
            w = _categorical(scores)
        elif r < gamma + alpha:
            # New source node w; pick existing target v by in-degree, biased to w's group.
            w = n_nodes
            d_in.append(0)
            d_out.append(0)
            g_w = int(jax.random.randint(next(key_seq), (), 0, k))
            group_members.setdefault(g_w, set()).add(w)
            groups[w] = g_w
            scores = _group_bonus(g_w, len(d_in)) * (np.asarray(d_in) + delta_in)
            v = _categorical(scores)
        else:
            # Edge between existing nodes: source w by out-degree, target v by in-degree in w's group.
            scores_w = np.asarray(d_out) + delta_out
            w = _categorical(scores_w)
            g_w = groups[w]
            scores_v = _group_bonus(g_w, len(d_in)) * (np.asarray(d_in) + delta_in)
            v = _categorical(scores_v)

        sources.append(w)
        targets.append(v)
        d_out[w] += 1
        d_in[v] += 1

    group_arr = np.array([groups[i] for i in range(n)], dtype=np.int32)
    return np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64), group_arr


def _key_stream(key: Array):
    """Yield an unbounded stream of distinct subkeys from ``key``."""
    while True:
        key, sub = jax.random.split(key)
        yield sub


def edges_to_grn(sources: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> GRN:
    """Collapse a directed multigraph edge list into a :class:`GRN`.

    Self-loops are dropped and parallel edges are merged into a single edge whose
    ``weight`` is the multiplicity (matching the reference's summed multigraph
    weights used for the interaction matrix).
    """
    num_genes = len(groups)
    mask = sources != targets
    src, tar = sources[mask], targets[mask]
    pair = src.astype(np.int64) * num_genes + tar.astype(np.int64)
    uniq, counts = np.unique(pair, return_counts=True)
    reg_idx = (uniq // num_genes).astype(np.int32)
    tar_idx = (uniq % num_genes).astype(np.int32)
    return GRN(
        reg_idx=jnp.asarray(reg_idx),
        tar_idx=jnp.asarray(tar_idx),
        weight=jnp.asarray(counts, dtype=jnp.float32),
        group=jnp.asarray(groups, dtype=jnp.int32),
    )


class GroupedScaleFreeSampler(GRNSampler):
    """Grouped scale-free GRN sampler (grn-paper).

    Defaults follow the grn-paper convention of parametrizing by ``r`` (the
    average regulators per gene): ``beta = 1 - 1/r``, ``gamma = 1/r``, ``alpha`` a
    tiny floor. Override any of ``alpha``/``beta``/``gamma`` directly to bypass this.
    """

    def __init__(
        self,
        r: float = 4.0,
        num_groups: int = 1,
        delta_in: float = 100.0,
        delta_out: float = 1.0,
        kappa: float = 10.0,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
    ) -> None:
        if alpha is None or beta is None or gamma is None:
            gamma = 1.0 / r
            beta = 1.0 - 1.0 / r
            alpha = max(1.0 - beta - gamma, 1e-12)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.num_groups = num_groups
        self.delta_in, self.delta_out, self.kappa = delta_in, delta_out, kappa

    def sample(self, key: Array, num_genes: int, **kwargs: object) -> GRN:
        params = {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta_in": self.delta_in,
            "delta_out": self.delta_out,
            "k": self.num_groups,
            "kappa": self.kappa,
        }
        params.update(kwargs)
        sources, targets, groups = grouped_scale_free_edges(key, num_genes, **params)
        return edges_to_grn(sources, targets, groups)
