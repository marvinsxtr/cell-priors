"""Shared helpers for the benchmark / comparison scripts."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from ..priors.sergio import SergioConfig, SergioPrior
from ..priors.sergio.grn import random_dag_edges


def build_prior(name: str, cfg: SergioConfig) -> SergioPrior:
    """Construct a prior by name (extension point for future priors)."""
    if name == "sergio":
        return SergioPrior(cfg)
    raise ValueError(f"Unknown prior '{name}'. Available: sergio")


def build_matched_networks(num_genes: int, num_cell_types: int, avg_regulators: float, seed: int):
    """Build a JAX :class:`SergioParams` and a structurally identical sergio_rs GRN.

    Used by the speed benchmark so both implementations simulate the same network
    topology and size. Returns ``(jax_params, sergio_grn, sergio_mr_profile)``.
    """
    import sergio_rs

    rng = np.random.default_rng(seed)
    reg_idx, tar_idx = random_dag_edges(rng, num_genes, avg_regulators)
    e = len(reg_idx)
    decay = rng.uniform(0.5, 1.0, num_genes)
    hill_n = rng.uniform(1.5, 2.5, e)
    k = rng.uniform(1.0, 5.0, e)

    high = rng.random((num_genes, num_cell_types)) < 0.5
    prod_rates = np.where(
        high,
        rng.uniform(3.0, 5.0, (num_genes, num_cell_types)),
        rng.uniform(0.5, 2.0, (num_genes, num_cell_types)),
    )
    from ..priors.sergio.grn import make_params

    jax_params = make_params(reg_idx, tar_idx, k, hill_n, decay, prod_rates)

    grn = sergio_rs.GRN()
    for (r, t), kk, nn in zip(zip(reg_idx.tolist(), tar_idx.tolist()), k, hill_n):
        grn.add_interaction(
            reg=sergio_rs.Gene(f"GENE{r:05d}", float(decay[r])),
            tar=sergio_rs.Gene(f"GENE{t:05d}", float(decay[t])),
            k=float(kk),
            h=None,
            n=int(round(nn)),
        )
    grn.set_mrs()
    mr_profile = sergio_rs.MrProfile.from_random(
        grn, num_cell_types=num_cell_types, low_range=(0.5, 2.0), high_range=(3.0, 5.0), seed=seed
    )
    return jax_params, grn, mr_profile


def timeit(fn: Callable[[], Any], repeats: int = 3) -> float:
    """Return the best wall-clock time (seconds) over ``repeats`` runs."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out) if _is_jax(out) else None
        best = min(best, time.perf_counter() - t0)
    return best


def _is_jax(x: Any) -> bool:
    return isinstance(x, jax.Array)
