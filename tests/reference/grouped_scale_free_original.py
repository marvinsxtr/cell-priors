"""Original grn-paper grouped scale-free graph sampler, copied for validation.

Adapted (only to return plain edge/group arrays instead of an ``nx.MultiDiGraph``,
so the test needs no networkx) from ``src/smallworld.py`` of
https://github.com/maguirre1/grn-paper -- the reference implementation for Aguirre
et al. 2025. The growth arithmetic (3-cycle seed, group seeding, the
alpha/beta/gamma moves and ``np.random.choice`` endpoint selection) is preserved
verbatim, and parallel edges / self-loops are kept exactly as the reference produces
them. See ``LICENSE`` in this directory (MIT, Copyright (c) 2023 maguirre1).
"""

from __future__ import annotations

import numpy as np


def grouped_scale_free_edges_reference(
    n,
    alpha=0.05,
    beta=0.54,
    gamma=0.41,
    delta_in=2,
    delta_out=0.0,
    k=None,
    kappa=1,
    seed=None,
):
    """Faithful copy of ``smallworld.grouped_scale_free_graph`` returning arrays.

    Returns ``(sources, targets, groups)``: the directed multigraph edge list (with
    self-loops and parallel edges kept, as the reference does) and a per-node group
    label. The only change from the reference is the return type (edge arrays + group
    labels instead of a constructed ``nx.MultiDiGraph``).
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")
    if beta < 0:
        raise ValueError("beta must be >= 0.")
    if gamma < 0:
        raise ValueError("gamma must be >= 0.")
    if abs(alpha + beta + gamma - 1.0) >= 1e-9:
        raise ValueError("alpha+beta+gamma must equal 1.")
    if delta_in < 0:
        raise ValueError("delta_in must be >= 0.")
    if delta_out < 0:
        raise ValueError("delta_out must be >= 0.")
    if kappa < 0:
        raise ValueError("kapppa must be >= 0")
    rng = np.random.default_rng(seed)
    if k is None:
        k = 1

    # Start with 3-cycle: (k-cycle massively drops modularity)
    V = {i for i in range(3)}
    if k < 3:
        if k == 2:
            K = {0: {0, 2}, 1: {1}}
        else:
            K = {0: {0, 1, 2}}
    else:
        K = {i: {i} for i in range(len(V))}
    E_s = list(V)
    E_t = [(i + 1) % len(V) for i in V]
    D = {"in": {i: E_t.count(i) for i in V}, "out": {i: E_s.count(i) for i in V}}

    while len(V) < n:
        r = rng.random()
        if r < gamma:
            # gamma: add new node v with random class k_v
            v = len(V)
            V.add(v)
            D["out"][v] = 0
            D["in"][v] = 0
            k_v = rng.choice(range(k))
            if k_v not in K:
                K[k_v] = {v}
            else:
                K[k_v].add(v)
            p = [(kappa if m in K[k_v] else 1) * (D["out"][m] + delta_out) for m in range(len(V))]
            w = rng.choice(list(range(len(V))), p=np.asarray(p) / np.sum(p))
        elif r < gamma + alpha:
            # alpha: add new node w with random class k_w
            w = len(V)
            V.add(w)
            D["out"][w] = 0
            D["in"][w] = 0
            k_w = rng.choice(range(k))
            if k_w not in K:
                K[k_w] = {w}
            else:
                K[k_w].add(w)
            q = [(kappa if m in K[k_w] else 1) * (D["in"][m] + delta_in) for m in range(len(V))]
            v = rng.choice(list(V), p=np.asarray(q) / np.sum(q))
        else:
            # beta: pick w by out degree, then v by in degree biased to w's group
            p = [(D["out"][m] + delta_out) for m in V]
            w = rng.choice(list(range(len(V))), p=np.asarray(p) / np.sum(p))
            k_w = [group for group, nodes in K.items() if w in nodes][0]
            q = [(kappa if m in K[k_w] else 1) * (D["in"][m] + delta_in) for m in range(len(V))]
            v = rng.choice(list(range(len(V))), p=np.asarray(q) / np.sum(q))
        E_s.append(int(w))
        E_t.append(int(v))
        D["out"][w] += 1
        D["in"][v] += 1

    groups = np.empty(n, dtype=np.int64)
    for group, nodes in K.items():
        for node in nodes:
            groups[node] = group
    return np.asarray(E_s, dtype=np.int64), np.asarray(E_t, dtype=np.int64), groups
