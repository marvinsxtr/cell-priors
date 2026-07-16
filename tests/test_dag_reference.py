"""Validate the jittable canonical DAG cycle removal against the host numpy reference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
from reference.dag_reference import dag_edge_mask_reference

from cell_priors.simulators.sergio.adapter import _acyclic_edge_mask
from cell_priors.simulators.sergio.dag import dag_edge_mask

_dag_jit = jax.jit(dag_edge_mask, static_argnums=(4,))


def _random_graph(seed: int, num_genes: int, num_edges: int, *, ties: bool, self_loops: bool):
    rng = np.random.default_rng(seed)
    reg = rng.integers(0, num_genes, size=num_edges)
    tar = rng.integers(0, num_genes, size=num_edges)
    if not self_loops:
        tar = np.where(tar == reg, (tar + 1) % num_genes, tar)
    weight = rng.integers(1, 5, size=num_edges).astype(np.float32) if ties else rng.uniform(0.1, 5.0, num_edges).astype(np.float32)
    active = rng.random(num_edges) < 0.85
    return reg.astype(np.int32), tar.astype(np.int32), weight, active


def _is_acyclic(reg, tar, keep, num_genes) -> bool:
    g = nx.DiGraph()
    g.add_nodes_from(range(num_genes))
    g.add_edges_from((int(r), int(t)) for r, t, k in zip(reg, tar, keep) if k)
    return nx.is_directed_acyclic_graph(g)


@pytest.mark.parametrize("seed", range(24))
@pytest.mark.parametrize("num_genes,num_edges", [(5, 12), (8, 20), (15, 45), (40, 80)])
@pytest.mark.parametrize("ties", [False, True])
def test_jax_matches_reference(seed, num_genes, num_edges, ties):
    reg, tar, weight, active = _random_graph(seed, num_genes, num_edges, ties=ties, self_loops=True)

    ref = dag_edge_mask_reference(reg, tar, weight, active, num_genes)
    got = np.asarray(_dag_jit(jnp.asarray(reg), jnp.asarray(tar), jnp.asarray(weight), jnp.asarray(active), num_genes))

    assert np.array_equal(got, ref)
    assert (got <= (active > 0)).all(), "kept edges must be a subset of the active edges"
    assert _is_acyclic(reg, tar, got, num_genes), "kept subgraph must be a DAG"


def test_disjoint_cycles_drop_the_weakest_edge():
    # cycle {0->1(3), 1->2(1), 2->0(2)} -> drop 1->2; cycle {3->4(5), 4->3(4)} -> drop 4->3;
    # acyclic edge 5->6(9) survives.
    reg = np.array([0, 1, 2, 3, 4, 5], np.int32)
    tar = np.array([1, 2, 0, 4, 3, 6], np.int32)
    weight = np.array([3.0, 1.0, 2.0, 5.0, 4.0, 9.0], np.float32)
    active = np.ones(6, bool)

    keep = dag_edge_mask_reference(reg, tar, weight, active, 7)

    assert keep.tolist() == [True, False, True, True, False, True]
    assert np.array_equal(np.asarray(_dag_jit(jnp.asarray(reg), jnp.asarray(tar), jnp.asarray(weight), jnp.asarray(active), 7)), keep)


@pytest.mark.parametrize("seed", range(12))
def test_matches_networkx_on_edge_disjoint_cycles(seed):
    # Edge-disjoint simple cycles with distinct weights: the greedy result is unique, so the
    # canonical algorithm must agree edge-for-edge with the host networkx cycle-breaker.
    rng = np.random.default_rng(seed)
    reg, tar, weight = [], [], []
    node = 0
    for _ in range(rng.integers(2, 5)):
        length = int(rng.integers(2, 5))
        cyc = list(range(node, node + length))
        node += length
        for i in range(length):
            reg.append(cyc[i])
            tar.append(cyc[(i + 1) % length])
    reg, tar = np.array(reg, np.int32), np.array(tar, np.int32)
    weight = (np.arange(len(reg)) + 1).astype(np.float32)  # all distinct
    rng.shuffle(weight)
    num_genes = node
    active = np.ones(len(reg), bool)

    canonical = dag_edge_mask_reference(reg, tar, weight, active, num_genes)
    host = _acyclic_edge_mask(reg, tar, weight, num_genes)

    assert np.array_equal(canonical, host)


def test_jit_vmap_batch_is_acyclic():
    batch, num_genes, num_edges = 8, 12, 30
    reg, tar, weight, active = zip(*(_random_graph(s, num_genes, num_edges, ties=False, self_loops=False) for s in range(batch)))
    fn = jax.jit(jax.vmap(dag_edge_mask, in_axes=(0, 0, 0, 0, None)), static_argnums=(4,))

    keep = np.asarray(fn(jnp.asarray(np.stack(reg)), jnp.asarray(np.stack(tar)), jnp.asarray(np.stack(weight)), jnp.asarray(np.stack(active)), num_genes))

    for b in range(batch):
        assert _is_acyclic(reg[b], tar[b], keep[b], num_genes)
