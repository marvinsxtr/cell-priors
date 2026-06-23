"""SERGIO as a :class:`Simulator` over the validated JAX core."""

from __future__ import annotations

import jax
from jax import Array

from ...base import GRN, InterventionKind, Simulator
from . import core, interventions
from .noise import NoiseProfile, add_technical_noise, resolve_profile
from .params import SergioConfig, SergioParams, build_sergio_params


class SergioSimulator(Simulator):
    """SERGIO stochastic-differential-equation expression simulator.

    ``cfg`` holds the static integration hyperparameters (cells, cell types,
    iterations, dt, noise); the remaining keyword arguments configure the kinetic
    parameters drawn when adapting a :class:`GRN`.

    ``acyclic`` selects how a sampled :class:`GRN` becomes parameters. ``True`` (default,
    strict SERGIO) removes cycles on the host so the topological steady state and master
    regulators apply. ``False`` keeps the graph as drawn and gives every gene a basal
    rate (cycle-tolerant), via a pure-JAX builder -- so structure sampling, kinetics and
    simulation compose inside a single ``jit``/``vmap``. Pair ``acyclic=False`` with
    ``SergioConfig(require_mrs=False)``.
    """

    def __init__(self, cfg: SergioConfig | None = None, acyclic: bool = True, **kinetics: object) -> None:
        self.cfg = cfg or SergioConfig()
        self.acyclic = acyclic
        self.kinetics = kinetics

    def build_params(self, grn: GRN, key: Array) -> SergioParams:
        c = self.cfg.num_cell_types
        if self.acyclic:
            from .adapter import grn_to_sergio_params

            return grn_to_sergio_params(grn, key, num_cell_types=c, acyclic=True, **self.kinetics)
        return build_sergio_params(grn, key, num_cell_types=c, **self.kinetics)

    def simulate(
        self,
        params: SergioParams,
        key: Array,
        add_noise: bool = False,
        noise_profile: str | NoiseProfile = "DS6",
    ) -> Array:
        """Simulate expression; ``noise_profile`` is a DS preset name or a custom
        :class:`NoiseProfile` (pass arbitrary technical-noise hyperparameters directly)."""
        k_sim, k_noise = jax.random.split(key)
        expr = core.simulate(params, k_sim, self.cfg)
        if add_noise:
            expr = add_technical_noise(k_noise, expr, resolve_profile(noise_profile))
        return expr

    def intervene(
        self,
        params: SergioParams,
        gene_indices: Array,
        kind: InterventionKind = InterventionKind.KNOCKOUT,
        strength: float = 1.0,
    ) -> SergioParams:
        kind = InterventionKind(kind)
        if kind is InterventionKind.KNOCKOUT:
            return interventions.knockout(params, gene_indices)
        return interventions.knockdown(params, gene_indices, strength=strength)
