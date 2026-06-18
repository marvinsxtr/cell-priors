"""The SERGIO prior: the uniform :class:`Prior` interface over the JAX core."""

from __future__ import annotations

from typing import Any

from jax import Array

from ...base import InterventionKind, Prior
from . import core, interventions
from .grn import SergioConfig, SergioParams, sample_random_params
from .noise import DS_PROFILES, add_technical_noise


class SergioPrior(Prior):
    """SERGIO single-cell expression prior.

    Wraps the pure functions in :mod:`core` with the uniform :class:`Prior` API.
    ``cfg`` holds the static simulation hyperparameters; pass it as a jit static
    argument when composing with a model.
    """

    def __init__(self, cfg: SergioConfig | None = None) -> None:
        self.cfg = cfg or SergioConfig()

    def sample_params(self, key: Array, num_genes: int = 100, **kwargs: Any) -> SergioParams:
        return sample_random_params(key, num_genes, num_cell_types=self.cfg.num_cell_types, **kwargs)

    def observational(
        self,
        params: SergioParams,
        key: Array,
        add_noise: bool = False,
        noise_profile: str = "DS6",
    ) -> Array:
        """Simulate expression; optionally add technical (sequencing) noise."""
        import jax

        k_sim, k_noise = jax.random.split(key)
        expr = core.simulate(params, k_sim, self.cfg)
        if add_noise:
            expr = add_technical_noise(k_noise, expr, DS_PROFILES[noise_profile])
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
