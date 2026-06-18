"""Shared helpers for the benchmark / comparison scripts."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from ..base import ComposedPrior, GRNSampler, Simulator
from ..samplers import GroupedScaleFreeSampler
from ..simulators.grn_paper import GrnPaperConfig, GrnPaperSimulator
from ..simulators.sergio import SergioConfig, SergioSimulator


def build_sampler(r: float = 3.0, num_groups: int = 1, kappa: float = 5.0) -> GRNSampler:
    """The grouped scale-free sampler (the only sampler)."""
    return GroupedScaleFreeSampler(r=r, num_groups=num_groups, kappa=kappa)


def build_simulator(name: str, **cfg_kwargs: Any) -> Simulator:
    """Construct a simulator by name."""
    if name == "sergio":
        return SergioSimulator(SergioConfig(**cfg_kwargs))
    if name == "grn_paper":
        return GrnPaperSimulator(GrnPaperConfig(**cfg_kwargs))
    raise ValueError(f"Unknown simulator '{name}'. Available: sergio, grn_paper")


def build_prior(simulator: str = "sergio", sampler_kwargs: dict | None = None, **cfg_kwargs: Any) -> ComposedPrior:
    """Compose the grouped scale-free sampler with a named simulator."""
    return ComposedPrior(build_sampler(**(sampler_kwargs or {})), build_simulator(simulator, **cfg_kwargs))


def matched_sergio_networks(num_genes: int, num_cell_types: int, sampler: GRNSampler, cfg: SergioConfig, seed: int):
    """Build a JAX :class:`SergioParams` and a structurally identical sergio_rs GRN.

    The sampler + SERGIO adapter produce the (acyclic) JAX network; the same final
    edge list, decay rates and Hill coefficients are replayed into sergio_rs so
    both implementations simulate the same topology. Returns
    ``(jax_params, sergio_grn, mr_profile)``.
    """
    import sergio_rs

    sim = SergioSimulator(cfg)
    key = jax.random.PRNGKey(seed)
    grn = sampler.sample(jax.random.fold_in(key, 1), num_genes)
    params = sim.build_params(grn, jax.random.fold_in(key, 2))

    reg = np.asarray(params.reg_idx)
    tar = np.asarray(params.tar_idx)
    k = np.asarray(params.k)
    hill_n = np.asarray(params.hill_n)
    decay = np.asarray(params.decay)

    rs_grn = sergio_rs.GRN()
    for (r, t), kk, nn in zip(zip(reg.tolist(), tar.tolist()), k, hill_n):
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
