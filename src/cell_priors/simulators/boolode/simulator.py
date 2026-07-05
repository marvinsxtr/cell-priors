"""BoolODE as a :class:`Simulator` over the validated JAX core."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ...base import GRN, InterventionKind, Simulator
from ..grn_paper.noise import maybe_add_noise
from ..sergio.noise import NoiseProfile
from . import core, interventions
from .core import K_DEFAULT, LP_DEFAULT, LX_DEFAULT, M_DEFAULT, N_DEFAULT, R_DEFAULT, BoolodeConfig, BoolodeParams


def make_params(
    reg_idx: np.ndarray,
    tar_idx: np.ndarray,
    is_act: np.ndarray,
    thr: np.ndarray,
    hill_n: np.ndarray,
    m: np.ndarray,
    l_x: np.ndarray,
    r: np.ndarray,
    l_p: np.ndarray,
    group: np.ndarray | None = None,
    dtype=jnp.float32,
) -> BoolodeParams:
    """Build neutral (un-intervened) :class:`BoolodeParams` from raw arrays."""
    e = len(reg_idx)
    g = len(m)
    return BoolodeParams(
        reg_idx=jnp.asarray(reg_idx, dtype=jnp.int32),
        tar_idx=jnp.asarray(tar_idx, dtype=jnp.int32),
        is_act=jnp.asarray(is_act, dtype=dtype),
        thr=jnp.asarray(thr, dtype=dtype),
        hill_n=jnp.asarray(hill_n, dtype=dtype),
        edge_mask=jnp.ones(e, dtype=dtype),
        m=jnp.asarray(m, dtype=dtype),
        l_x=jnp.asarray(l_x, dtype=dtype),
        r=jnp.asarray(r, dtype=dtype),
        l_p=jnp.asarray(l_p, dtype=dtype),
        prod_scale=jnp.ones(g, dtype=dtype),
        ko_mask=jnp.zeros(g, dtype=dtype),
        group=jnp.zeros(g, dtype=jnp.int32) if group is None else jnp.asarray(group, dtype=jnp.int32),
    )


def build_boolode_params(
    grn: GRN,
    key: Array,
    repression_prob_range: tuple[float, float] = (0.0, 0.5),
    param_jitter: float = 0.1,
    dtype=jnp.float32,
) -> BoolodeParams:
    """Sample BoolODE kinetics for a :class:`GRN` as a pure JAX function.

    Keeps the graph exactly as drawn (a unique edge is active where ``weight > 0``;
    duplicate / padding / self-loop slots are masked out). Each edge is a repressor
    with a per-network probability drawn from ``repression_prob_range`` (mirroring
    SERGIO), else an activator. Kinetic rates are BoolODE's defaults perturbed by
    ``+-param_jitter`` (BoolODE's ``sample_pars`` regime). Only uses ``jax.random``
    with fixed shapes, so it composes with the sampler + simulator in one
    ``jit``/``vmap``.
    """
    e = grn.num_edges
    g = grn.num_genes
    k_rep, k_sign, k_thr, k_n, k_m, k_lx, k_r, k_lp = jax.random.split(key, 8)

    def _jit(k, shape, base):
        lo, hi = base * (1.0 - param_jitter), base * (1.0 + param_jitter)
        return jax.random.uniform(k, shape, minval=lo, maxval=hi)

    repression_prob = jax.random.uniform(k_rep, (), minval=repression_prob_range[0], maxval=repression_prob_range[1])
    is_act = (jax.random.uniform(k_sign, (e,)) >= repression_prob).astype(dtype)
    return BoolodeParams(
        reg_idx=grn.reg_idx.astype(jnp.int32),
        tar_idx=grn.tar_idx.astype(jnp.int32),
        is_act=is_act,
        thr=_jit(k_thr, (e,), K_DEFAULT).astype(dtype),
        hill_n=_jit(k_n, (e,), N_DEFAULT).astype(dtype),
        edge_mask=(grn.weight > 0).astype(dtype),
        m=_jit(k_m, (g,), M_DEFAULT).astype(dtype),
        l_x=_jit(k_lx, (g,), LX_DEFAULT).astype(dtype),
        r=_jit(k_r, (g,), R_DEFAULT).astype(dtype),
        l_p=_jit(k_lp, (g,), LP_DEFAULT).astype(dtype),
        prod_scale=jnp.ones(g, dtype=dtype),
        ko_mask=jnp.zeros(g, dtype=dtype),
        group=grn.group.astype(jnp.int32),
    )


class BoolodeSimulator(Simulator):
    """BoolODE chemical-Langevin expression simulator (Pratapa et al. 2020).

    Each cell is an independent stochastic trajectory of the mRNA/protein ODE
    system, observed at a random post-burn-in time point. Regulation follows the
    canonical Boolean rule ``(OR activators) AND NOT (OR repressors)``; a hard
    knockout silences a gene and drops its outgoing edges, a soft knockdown
    attenuates its transcription with the graph intact.
    """

    def __init__(self, cfg: BoolodeConfig | None = None, **kinetics: object) -> None:
        self.cfg = cfg or BoolodeConfig()
        self.kinetics = kinetics

    def build_params(self, grn: GRN, key: Array) -> BoolodeParams:
        return build_boolode_params(grn, key, **self.kinetics)

    def simulate(
        self,
        params: BoolodeParams,
        key: Array,
        add_noise: bool = False,
        noise_profile: str | NoiseProfile = "DS6",
    ) -> Array:
        k_sim, k_noise = jax.random.split(key)
        expr = core.simulate(params, k_sim, self.cfg)
        return maybe_add_noise(expr, k_noise, add_noise, noise_profile)

    def intervene(
        self,
        params: BoolodeParams,
        gene_indices: Array,
        kind: InterventionKind = InterventionKind.KNOCKOUT,
        strength: float = 1.0,
    ) -> BoolodeParams:
        kind = InterventionKind(kind)
        if kind is InterventionKind.KNOCKOUT:
            return interventions.knockout(params, gene_indices)
        return interventions.knockdown(params, gene_indices, strength=strength)
