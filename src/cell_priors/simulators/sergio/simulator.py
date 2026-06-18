"""SERGIO as a :class:`Simulator` over the validated JAX core."""

from __future__ import annotations

import jax
from jax import Array

from ...base import GRN, InterventionKind, Simulator
from . import core, interventions
from .adapter import grn_to_sergio_params
from .noise import NoiseProfile, add_technical_noise, resolve_profile
from .params import SergioConfig, SergioParams


class SergioSimulator(Simulator):
    """SERGIO stochastic-differential-equation expression simulator.

    ``cfg`` holds the static integration hyperparameters (cells, cell types,
    iterations, dt, noise); the remaining keyword arguments configure the kinetic
    parameters drawn when adapting a :class:`GRN`.
    """

    def __init__(self, cfg: SergioConfig | None = None, **kinetics: object) -> None:
        self.cfg = cfg or SergioConfig()
        self.kinetics = kinetics

    def build_params(self, grn: GRN, key: Array) -> SergioParams:
        return grn_to_sergio_params(grn, key, num_cell_types=self.cfg.num_cell_types, **self.kinetics)

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
