"""Shared helpers for the benchmark / comparison scripts."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from ..base import ComposedPrior, GRNSampler, Simulator
from ..samplers import GroupedScaleFreeSampler
from ..simulators.boolode import BoolodeConfig, BoolodeSimulator
from ..simulators.grn_paper import GrnPaperConfig, GrnPaperSimulator
from ..simulators.sergio import SergioConfig, SergioSimulator, make_params


def build_sampler(r: float = 3.0, num_groups: int = 1, kappa: float = 5.0) -> GRNSampler:
    """The grouped scale-free sampler (the only sampler)."""
    return GroupedScaleFreeSampler(r=r, num_groups=num_groups, kappa=kappa)


def build_simulator(name: str, **cfg_kwargs: Any) -> Simulator:
    """Construct a simulator by name."""
    if name == "sergio":
        return SergioSimulator(SergioConfig(**cfg_kwargs))
    if name == "grn_paper":
        return GrnPaperSimulator(GrnPaperConfig(**cfg_kwargs))
    if name == "boolode":
        return BoolodeSimulator(BoolodeConfig(**cfg_kwargs))
    raise ValueError(f"Unknown simulator '{name}'. Available: sergio, grn_paper, boolode")


def build_prior(simulator: str = "sergio", sampler_kwargs: dict | None = None, **cfg_kwargs: Any) -> ComposedPrior:
    """Compose the grouped scale-free sampler with a named prior/simulator.

    ``mappfn`` is the cycle-tolerant SERGIO prior (any GRN, no required master
    regulators); ``sergio`` and ``grn_paper`` are the strict simulators.
    """
    from ..priors import MapPFNPrior

    sampler = build_sampler(**(sampler_kwargs or {}))
    if simulator == "mappfn":
        return MapPFNPrior(SergioConfig(**cfg_kwargs), sampler=sampler)
    return ComposedPrior(sampler, build_simulator(simulator, **cfg_kwargs))


def matched_sergio_networks(
    num_genes: int, num_cell_types: int, cfg: SergioConfig, seed: int, avg_regulators: float = 3.0
):
    """Build a random DAG and a matching JAX :class:`SergioParams` + sergio_rs GRN.

    Edges only go from a lower to a higher gene index, so the graph is acyclic and gene 0
    is a master regulator -- exactly what strict SERGIO (and ``sergio_rs``) require, with no
    host-side cycle breaking. The same edges, decay rates and Hill coefficients drive both
    backends. Returns ``(jax_params, sergio_grn, mr_profile)``.
    """
    import sergio_rs

    rng = np.random.default_rng(seed)
    # Each gene j >= 1 draws a few regulators from {0..j-1}; gene 0 is the source (MR).
    reg, tar = [], []
    for j in range(1, num_genes):
        n_reg = min(j, 1 + rng.poisson(max(avg_regulators - 1.0, 0.0)))
        for r in rng.choice(j, size=n_reg, replace=False):
            reg.append(int(r))
            tar.append(j)
    reg = np.asarray(reg)
    tar = np.asarray(tar)

    decay = rng.uniform(0.5, 1.0, num_genes)
    k = rng.uniform(1.0, 5.0, len(reg))
    hill_n = rng.uniform(1.5, 2.5, len(reg))
    is_mr = np.bincount(tar, minlength=num_genes) == 0
    prod_rates = np.where(
        rng.random((num_genes, num_cell_types)) < 0.5,
        rng.uniform(3.0, 5.0, (num_genes, num_cell_types)),
        rng.uniform(0.5, 2.0, (num_genes, num_cell_types)),
    )
    prod_rates = prod_rates * is_mr[:, None]  # only master regulators carry basal drive
    params = make_params(reg, tar, k, hill_n, decay, prod_rates)

    rs_grn = sergio_rs.GRN()
    for r, t, kk, nn in zip(reg.tolist(), tar.tolist(), k, hill_n):
        rs_grn.add_interaction(
            reg=sergio_rs.Gene(f"GENE{r:05d}", float(decay[r])),
            tar=sergio_rs.Gene(f"GENE{t:05d}", float(decay[t])),
            k=float(kk),
            h=None,
            n=int(round(float(nn))),
        )
    rs_grn.set_mrs()
    mr_profile = sergio_rs.MrProfile.from_random(
        rs_grn, num_cell_types=num_cell_types, low_range=(0.5, 2.0), high_range=(3.0, 5.0), seed=seed
    )
    return params, rs_grn, mr_profile


def timeit(fn: Callable[[], Any], repeats: int = 3) -> float:
    """Return the best wall-clock time (seconds) over ``repeats`` runs."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, jax.Array):
            jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
    return best
