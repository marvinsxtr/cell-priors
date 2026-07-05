"""Pure-JAX core of the BoolODE expression model (Pratapa et al. 2020).

BoolODE turns a gene-regulatory network into a system of chemical Langevin
equations. Each gene ``g`` has two species -- an mRNA ``x_g`` and a protein
``p_g`` -- and regulation acts through the *protein* levels::

    dx_g/dt = m_g * f_g(p)      - l_x_g * x_g
    dp_g/dt = r_g * x_g         - l_p_g * p_g

``f_g in [0, 1]`` is the transcriptional activation function. In BoolODE it is a
ratio of sums of products of Hill terms whose coefficients come from the truth
table of the gene's Boolean rule. We reimplement the standard rule

    gene ON  <=>  (any activator present) AND NOT (any repressor present)

(the canonical choice when only a *signed* network is known, not full Boolean
logic), for which the combinatorial BoolODE expansion collapses to a closed form.

With per-edge Hill term ``H_e = (p[reg_e]/k_e)^n_e`` and, per target gene ``g``,

    P_A[g]   = prod over active *activator*  edges of (1 + H_e)
    P_all[g] = prod over all active          edges of (1 + H_e)

BoolODE's activation function is exactly

    f_g = (basal_g + P_A[g] - 1) / P_all[g],

where ``basal_g = 1`` iff ``g`` has no active incoming activator edge (a source, or
a purely-repressed gene, is constitutively expressed and repressible -- the
analogue of a SERGIO master regulator). This is numerically identical to BoolODE's
``alpha_0 + sum_C a_C prod H`` over regulator combinations ``C`` for the rule above,
but costs ``O(E)`` per step via two ``segment_sum``s instead of ``O(2^indeg)``.

Everything is a pure function of an array-valued :class:`BoolodeParams`, so it is
``jit``/``vmap``/``scan``-able and fuses with a JAX model in one graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array, lax, random


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BoolodeParams:
    """Parameters of a BoolODE model (a pytree of arrays).

    Edges are directed ``reg -> tar`` with a sign (``is_act``: 1 activator, 0
    repressor) and a per-edge Hill threshold/coefficient. ``edge_mask``,
    ``prod_scale`` and ``ko_mask`` are intervention knobs (neutral by default) so
    an intervened network is just another ``BoolodeParams`` with identical shapes.
    """

    # Per-edge quantities, shape (E,)
    reg_idx: Array  # int: regulator gene index
    tar_idx: Array  # int: target gene index
    is_act: Array  # float {0,1}: 1 activator, 0 repressor
    thr: Array  # float: Hill half-response threshold k
    hill_n: Array  # float: Hill coefficient n
    edge_mask: Array  # float {0,1}: 0 removes the edge (hard knockout)

    # Per-gene quantities, shape (G,)
    m: Array  # float: mRNA transcription rate
    l_x: Array  # float: mRNA degradation rate
    r: Array  # float: protein translation rate
    l_p: Array  # float: protein degradation rate
    prod_scale: Array  # float: transcription multiplier (soft CRISPRi knockdown)
    ko_mask: Array  # float {0,1}: 1 forces the gene's mRNA + protein to zero
    group: Array  # int: module label (carried through from the GRN)

    @property
    def num_genes(self) -> int:
        return self.m.shape[0]

    @property
    def num_edges(self) -> int:
        return self.reg_idx.shape[0]


@dataclass(frozen=True)
class BoolodeConfig:
    """Static integration hyperparameters (passed as a jit static argument).

    Defaults follow BoolODE: ``dt`` is the integration step, ``noise_c`` the
    Chemical-Langevin noise amplitude (BoolODE's ``c = 10``). Each cell is an
    independent stochastic trajectory sampled at one random post-burn-in step.
    """

    num_cells: int = 100
    n_steps: int = 2000
    burnin: int = 1000
    dt: float = 0.01
    noise_c: float = 10.0

    def __post_init__(self) -> None:
        if not 0 <= self.burnin < self.n_steps:
            raise ValueError("require 0 <= burnin < n_steps")


# BoolODE default kinetic parameters (parameters.yaml).
M_DEFAULT = 20.0  # mRNATranscription
LX_DEFAULT = 10.0  # mRNADegradation
R_DEFAULT = 10.0  # proteinTranslation
LP_DEFAULT = 1.0  # proteinDegradation
N_DEFAULT = 10.0  # hillCoefficient
# hillThreshold = y_max / 2, y_max = (m/l_x) * (r/l_p) = 2 * 10 = 20 -> k = 10.
K_DEFAULT = (M_DEFAULT / LX_DEFAULT) * (R_DEFAULT / LP_DEFAULT) / 2.0


def _hill(p: Array, prm: BoolodeParams) -> Array:
    """Per-edge Hill term ``(p[reg]/thr)^n`` (activity of each regulator), shape ``(E,)``."""
    return (jnp.maximum(p[prm.reg_idx], 0.0) / prm.thr) ** prm.hill_n


def activation(p: Array, prm: BoolodeParams) -> Array:
    """BoolODE transcriptional activation ``f_g(p) in [0, 1]``, shape ``(G,)``.

    Closed form of the canonical rule ``(OR activators) AND NOT (OR repressors)``
    with constitutive basal expression for genes lacking an active activator.
    """
    g = prm.num_genes
    h = _hill(p, prm) * prm.edge_mask  # inactive edges contribute H = 0 (factor 1)
    log1p_h = jnp.log1p(h)
    # Products via exp(sum(log(1 + H))); masked edges add 0 -> multiply by 1.
    log_p_all = jax.ops.segment_sum(log1p_h, prm.tar_idx, num_segments=g)
    log_p_act = jax.ops.segment_sum(log1p_h * prm.is_act, prm.tar_idx, num_segments=g)
    p_all = jnp.exp(log_p_all)
    p_act = jnp.exp(log_p_act)
    # A gene with no active incoming activator edge is constitutively expressed.
    act_indeg = jax.ops.segment_sum(prm.is_act * prm.edge_mask, prm.tar_idx, num_segments=g)
    basal = (act_indeg == 0).astype(p.dtype)
    return (basal + p_act - 1.0) / p_all


def _drift(x: Array, p: Array, prm: BoolodeParams) -> tuple[Array, Array]:
    """Time derivatives ``(dx, dp)`` of the deterministic BoolODE system."""
    f = activation(p, prm)
    dx = prm.prod_scale * prm.m * f - prm.l_x * x
    dp = prm.r * x - prm.l_p * p
    return dx, dp


def _euler_step(x: Array, p: Array, prm: BoolodeParams, cfg: BoolodeConfig, key: Array | None) -> tuple[Array, Array]:
    """One Euler(-Maruyama) step. ``key=None`` gives the deterministic ODE step.

    Matches BoolODE's integrator: Chemical-Langevin noise ``c*sqrt(|y|)*dW`` with
    ``dW ~ N(0, dt)``, and a positivity rule that *reverts* a species to its
    previous value if a step would take it negative (rather than clipping to 0).
    """
    dt = cfg.dt
    dx, dp = _drift(x, p, prm)
    x_new = x + dx * dt
    p_new = p + dp * dt
    if key is not None:
        kx, kp = random.split(key)
        dw_x = random.normal(kx, x.shape) * dt
        dw_p = random.normal(kp, p.shape) * dt
        x_new = x_new + cfg.noise_c * jnp.sqrt(jnp.abs(x)) * dw_x
        p_new = p_new + cfg.noise_c * jnp.sqrt(jnp.abs(p)) * dw_p
    x_new = jnp.where(x_new < 0, x, x_new)
    p_new = jnp.where(p_new < 0, p, p_new)
    ko = prm.ko_mask
    return x_new * (1.0 - ko), p_new * (1.0 - ko)


def initial_state(prm: BoolodeParams) -> tuple[Array, Array]:
    """BoolODE initial condition: ``x = 1`` and ``p = (r/l_p) * x`` per gene."""
    x0 = jnp.ones(prm.num_genes, dtype=prm.m.dtype)
    p0 = (prm.r / prm.l_p) * x0
    return x0 * (1.0 - prm.ko_mask), p0 * (1.0 - prm.ko_mask)


def steady_state(prm: BoolodeParams, cfg: BoolodeConfig) -> tuple[Array, Array]:
    """Integrate the deterministic ODE to its steady state, returning ``(x, p)``."""

    def step(carry, _):
        x, p = carry
        return _euler_step(x, p, prm, cfg, None), None

    (x, p), _ = lax.scan(step, initial_state(prm), None, length=cfg.n_steps)
    return x, p


def simulate(prm: BoolodeParams, key: Array, cfg: BoolodeConfig) -> Array:
    """Simulate ``num_cells`` independent trajectories; return mRNA, shape ``(cells, G)``.

    Each cell integrates the Chemical-Langevin SDE and is observed at a single
    random post-burn-in step -- BoolODE's "one cell = one time point of one
    simulation" sampling. Genes (mRNA) are reported; proteins are latent.
    """
    g = prm.num_genes
    k_cells, k_time = random.split(key)
    # One random observation step per cell in the post-burn-in window.
    obs_step = random.randint(k_time, (cfg.num_cells,), cfg.burnin, cfg.n_steps)

    def one_cell(cell_key, obs):
        def step(carry, t):
            x, p, captured = carry
            x, p = _euler_step(x, p, prm, cfg, random.fold_in(cell_key, t))
            captured = jnp.where(t == obs, x, captured)
            return (x, p, captured), None

        x0, p0 = initial_state(prm)
        (_, _, captured), _ = lax.scan(step, (x0, p0, x0), jnp.arange(cfg.n_steps))
        return captured

    return jax.vmap(one_cell)(random.split(k_cells, cfg.num_cells), obs_step).reshape(cfg.num_cells, g)
