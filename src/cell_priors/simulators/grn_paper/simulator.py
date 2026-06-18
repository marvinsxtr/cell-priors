"""The grn-paper model as a :class:`Simulator`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ...base import GRN, InterventionKind, Simulator
from . import core
from .core import GrnPaperConfig, GrnPaperParams
from .noise import maybe_add_noise


def build_grn_paper_params(grn: GRN, key: Array, inflate_edges: bool = True, dtype=jnp.float32) -> GrnPaperParams:
    """Attach grn-paper expression parameters to a :class:`GRN` (pure JAX).

    Mirrors ``add_expression_parameters``: the dense interaction matrix is
    ``beta = S * E`` with ``S ~ Normal(0, 1)`` (inflated by its sign so |weight| is
    pushed away from 0) and ``E`` the integer edge-multiplicity matrix; ``alpha`` is
    ``logit(Beta(2, 8))`` and ``l = max(sigmoid(-alpha), Beta(8, 2))``.
    """
    g = grn.num_genes
    k_s, k_a, k_l = jax.random.split(key, 3)

    e_mat = jnp.zeros((g, g), dtype=dtype).at[grn.reg_idx, grn.tar_idx].add(grn.weight.astype(dtype))
    s = jax.random.normal(k_s, (g, g), dtype=dtype)
    if inflate_edges:
        s = s + jnp.sign(s)
    beta = s * e_mat

    alpha = jax.scipy.special.logit(jax.random.beta(k_a, 2.0, 8.0, (g,)).astype(dtype))
    l = jnp.maximum(jax.nn.sigmoid(-alpha), jax.random.beta(k_l, 8.0, 2.0, (g,)).astype(dtype))
    return GrnPaperParams(beta=beta, alpha=alpha, l=l, group=grn.group)


class GrnPaperSimulator(Simulator):
    """Sigmoid-link SDE expression simulator from Aguirre et al. 2025.

    Each cell is an independent SDE realization observed as its post-burn-in time
    average. Interventions act on a gene's *outgoing* interactions: a hard knockout
    zeros them; a soft knockdown attenuates them by ``strength``.
    """

    def __init__(self, cfg: GrnPaperConfig | None = None, inflate_edges: bool = True) -> None:
        self.cfg = cfg or GrnPaperConfig()
        self.inflate_edges = inflate_edges

    def build_params(self, grn: GRN, key: Array) -> GrnPaperParams:
        return build_grn_paper_params(grn, key, inflate_edges=self.inflate_edges)

    def simulate(
        self,
        params: GrnPaperParams,
        key: Array,
        add_noise: bool = False,
        noise_profile: str = "DS6",
    ) -> Array:
        k_sim, k_noise = jax.random.split(key)
        expr = core.simulate(params, k_sim, self.cfg)
        return maybe_add_noise(expr, k_noise, add_noise, noise_profile)

    def intervene(
        self,
        params: GrnPaperParams,
        gene_indices: Array,
        kind: InterventionKind = InterventionKind.KNOCKOUT,
        strength: float = 1.0,
    ) -> GrnPaperParams:
        kind = InterventionKind(kind)
        if kind is InterventionKind.KNOCKOUT:
            return core.knockout(params, gene_indices)
        return core.knockdown(params, gene_indices, strength=strength)
