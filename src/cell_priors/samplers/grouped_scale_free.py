"""Grouped scale-free GRN sampler (Aguirre et al. 2025), as a single jitted scan.

A JAX port of ``grouped_scale_free_graph`` from the grn-paper. The graph grows by
Bollobas-style directed preferential attachment with three moves (probabilities
``alpha``/``beta``/``gamma`` summing to 1) plus group structure: ``k`` modules with
within-group preferential attachment controlled by ``kappa``.

The growth is sequential -- each edge's endpoint distribution depends on the
degrees accumulated so far -- but it is expressed as a fixed-length ``lax.scan``
over ``(num_genes,)`` degree buffers rather than a host Python loop, so the whole
sampler is one ``jit``/``vmap``-able function with no host round-trips.

To get a static output shape, edges are stored in a fixed ``max_edges`` buffer with
a per-edge validity mask. The graph is seeded with a 3-cycle and then each scan
iteration adds one edge; a ``gamma``/``alpha`` move also adds a node, a ``beta``
move only adds an edge. The walk runs until ``num_genes`` nodes exist and then
emits masked (inert) padding edges for the remaining iterations, so a single
``max_edges`` covers the (random) true edge count. ``max_edges`` is chosen from the
node-addition probability with a wide safety margin.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array, lax

from ..base import GRN, GRNSampler


def default_max_edges(num_genes: int, alpha: float, gamma: float) -> int:
    """A safe static upper bound on the number of edges grown for ``num_genes`` nodes.

    Each iteration adds a node with probability ``p = alpha + gamma`` and an edge
    every time; the iteration count to reach ``num_genes`` nodes is negative-binomial
    with mean ``(num_genes - 3) / p``. The bound is the mean plus a wide margin so the
    walk almost surely completes within it (any shortfall just yields fewer nodes).
    """
    p = float(alpha) + float(gamma)
    if p <= 0.0:
        raise ValueError("alpha + gamma must be > 0 (otherwise no nodes are ever added).")
    remaining = max(num_genes - 3, 0)
    mean = remaining / p
    std = math.sqrt(remaining * (1.0 - p)) / p
    return int(3 + math.ceil(mean + 8.0 * std) + 16)


def _seed_groups(k: int) -> list[int]:
    """Group labels of the 3-cycle seed nodes (matches the reference seeding)."""
    if k == 1:
        return [0, 0, 0]
    if k == 2:
        return [0, 1, 0]
    return [0, 1, 2]


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
    max_edges: int | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Grow a grouped scale-free directed multigraph as a jitted scan.

    Returns ``(sources, targets, edge_valid, groups)``: ``sources``/``targets`` are
    ``(max_edges,)`` int arrays, ``edge_valid`` a ``(max_edges,)`` float mask (1 for a
    real edge, 0 for a self-loop or post-completion padding edge), and ``groups`` a
    ``(n,)`` module label per node. ``n`` and ``k`` are static (they set shapes and the
    seed grouping). Endpoint selection uses ``categorical`` over log attachment scores,
    the same distribution as the reference's ``choice(p=...)``.
    """
    if abs(alpha + beta + gamma - 1.0) >= 1e-9:
        raise ValueError("alpha + beta + gamma must equal 1.")
    for name, val in (("alpha", alpha), ("beta", beta), ("gamma", gamma),
                      ("delta_in", delta_in), ("delta_out", delta_out), ("kappa", kappa)):
        if val < 0:
            raise ValueError(f"{name} must be >= 0.")
    if n < 3:
        raise ValueError("num_genes must be >= 3 (the seed is a 3-cycle).")
    k = max(int(k), 1)
    max_edges = default_max_edges(n, alpha, gamma) if max_edges is None else int(max_edges)
    if max_edges < n:
        raise ValueError("max_edges must be >= num_genes.")

    node_idx = jnp.arange(n)
    seed_nodes = jnp.array([0, 1, 2])

    d_in0 = jnp.zeros(n, jnp.int32).at[seed_nodes].set(1)
    d_out0 = jnp.zeros(n, jnp.int32).at[seed_nodes].set(1)
    group0 = jnp.full(n, -1, jnp.int32).at[seed_nodes].set(jnp.asarray(_seed_groups(k), jnp.int32))
    cur0 = jnp.int32(3)

    def _pick(pick_key: Array, scores: Array, valid: Array) -> Array:
        logits = jnp.where(valid, jnp.log(jnp.maximum(scores, 0.0)), -jnp.inf)
        return jax.random.categorical(pick_key, logits)

    def _add_node(group: Array, idx: Array, g_new: Array, n_after: Array):
        group = group.at[idx].set(g_new)
        valid = node_idx < n_after
        return group, valid

    def gamma_move(d_in, d_out, group, cur, g_new, kw, kv):
        # New target node v=cur; pick existing source w by out-degree, biased to v's group.
        group, valid = _add_node(group, cur, g_new, cur + 1)
        bonus = jnp.where(group == g_new, kappa, 1.0)
        w = _pick(kw, bonus * (d_out.astype(jnp.float32) + delta_out), valid)
        return w, cur, group, cur + 1

    def alpha_move(d_in, d_out, group, cur, g_new, kw, kv):
        # New source node w=cur; pick existing target v by in-degree, biased to w's group.
        group, valid = _add_node(group, cur, g_new, cur + 1)
        bonus = jnp.where(group == g_new, kappa, 1.0)
        v = _pick(kv, bonus * (d_in.astype(jnp.float32) + delta_in), valid)
        return cur, v, group, cur + 1

    def beta_move(d_in, d_out, group, cur, g_new, kw, kv):
        # Edge between existing nodes: source w by out-degree, target v by in-degree in w's group.
        valid = node_idx < cur
        w = _pick(kw, d_out.astype(jnp.float32) + delta_out, valid)
        bonus = jnp.where(group == group[w], kappa, 1.0)
        v = _pick(kv, bonus * (d_in.astype(jnp.float32) + delta_in), valid)
        return w, v, group, cur

    def step(carry, step_key):
        d_in, d_out, group, cur = carry
        km, kg, kw, kv = jax.random.split(step_key, 4)
        r = jax.random.uniform(km)
        move = jnp.where(r < gamma, 0, jnp.where(r < gamma + alpha, 1, 2))
        g_new = jax.random.randint(kg, (), 0, k)

        def active_step(_):
            w, v, group_new, cur_new = lax.switch(
                move, [gamma_move, alpha_move, beta_move], d_in, d_out, group, cur, g_new, kw, kv
            )
            carry_new = (d_in.at[v].add(1), d_out.at[w].add(1), group_new, cur_new)
            return carry_new, (w, v, 1.0)

        def done_step(_):
            return carry, (jnp.int32(0), jnp.int32(0), 0.0)

        return lax.cond(cur < n, active_step, done_step, operand=None)

    step_keys = jax.random.split(key, max_edges - 3)
    (d_in, d_out, group, cur), (grown_s, grown_t, grown_valid) = lax.scan(
        step, (d_in0, d_out0, group0, cur0), step_keys
    )

    sources = jnp.concatenate([seed_nodes, grown_s])
    targets = jnp.concatenate([jnp.array([1, 2, 0]), grown_t])
    valid = jnp.concatenate([jnp.ones(3, jnp.float32), grown_valid])
    valid = valid * (sources != targets).astype(jnp.float32)  # drop self-loops (reference drops them)
    group = jnp.where(group < 0, 0, group)
    return sources.astype(jnp.int32), targets.astype(jnp.int32), valid, group


def edges_to_grn(sources: Array, targets: Array, edge_valid: Array, groups: Array) -> GRN:
    """Wrap a fixed-size masked edge list as a :class:`GRN`, collapsing parallel edges.

    The directed multigraph can contain the same ``reg -> tar`` pair several times. To
    match the grn-paper convention (the interaction matrix sums parallel edges), each
    unique edge is represented once -- on its first occurrence, carrying ``weight`` equal
    to the pair's multiplicity -- and the remaining rows (duplicates, self-loops, padding)
    get ``weight = 0``. The shape stays fixed, so this is a pure ``jit``-able function; it
    uses two ``(num_genes, num_genes)`` scatters (O(E) work) rather than a quadratic scan.
    """
    n = groups.shape[0]
    e = sources.shape[0]
    valid = (edge_valid > 0).astype(jnp.float32)

    multiplicity = jnp.zeros((n, n), jnp.float32).at[sources, targets].add(valid)
    order = jnp.where(valid > 0, jnp.arange(e), e).astype(jnp.int32)
    first = jnp.full((n, n), e, jnp.int32).at[sources, targets].min(order)
    is_representative = (valid > 0) & (jnp.arange(e) == first[sources, targets])
    weight = jnp.where(is_representative, multiplicity[sources, targets], 0.0)

    return GRN(
        reg_idx=sources.astype(jnp.int32),
        tar_idx=targets.astype(jnp.int32),
        weight=weight.astype(jnp.float32),
        group=groups.astype(jnp.int32),
    )


class GroupedScaleFreeSampler(GRNSampler):
    """Grouped scale-free GRN sampler (grn-paper).

    Defaults follow the grn-paper convention of parametrizing by ``r`` (the average
    regulators per gene): ``beta = 1 - 1/r``, ``gamma = 1/r``, ``alpha`` a tiny floor.
    Override any of ``alpha``/``beta``/``gamma`` directly to bypass this.
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

    def sample(self, key: Array, num_genes: int, max_edges: int | None = None, **kwargs: object) -> GRN:
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
        sources, targets, valid, groups = grouped_scale_free_edges(key, num_genes, max_edges=max_edges, **params)
        return edges_to_grn(sources, targets, valid, groups)
