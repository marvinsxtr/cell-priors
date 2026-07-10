"""Pure-JAX SERGIO simulation core.

This is a faithful, vectorized reimplementation of the SERGIO v2 stochastic
differential equation simulator (as in ``sergio_rs``). Everything here is a pure
function of array-valued :class:`SergioParams`, so it can be ``jit``/``vmap``/
``scan``-ed and composed with a JAX model in a single graph without leaving the
device.

Speed-oriented design choices (cf. the purejaxrl philosophy of folding the whole
environment into the JAX graph):

* The regulatory interactions are a *sparse* edge list, so each step costs
  ``O(E * C)`` rather than ``O(G**2 * C)`` (``G`` genes, ``E`` edges, ``C`` cell
  types). Edge contributions are scattered to target genes with a single
  ``segment_sum``.
* The expensive part of SERGIO -- estimating per-edge half-responses and the
  steady state, which the reference does with a sequential pass over topological
  levels -- is recast as a fixed-point iteration. On a DAG it converges to the
  exact same values in at most ``depth`` iterations, but it is a plain ``scan``
  with no Python-level graph traversal, so it stays inside ``jit``.
* The whole time integration is a single ``lax.scan`` over the (static) number of
  iterations; the trajectory is stacked once and sampled with a gather.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax import Array, lax, random

from .params import SergioConfig, SergioParams


def _edge_production(x: Array, p: SergioParams) -> Array:
    """Hill production contributed to each gene by its incoming edges.

    ``x`` is the current concentration, shape ``(G, C)``. Returns shape ``(G, C)``.
    For each edge the Hill term is ``x_reg^n / (x_reg^n + h^n)``; activating edges
    (``k > 0``) contribute ``k * hill`` and repressing edges (``k < 0``) contribute
    ``|k| * (1 - hill)``, exactly as in SERGIO's ``Interaction::get_hill``.
    """
    g = p.num_genes
    xr = x[p.reg_idx]  # (E, C)
    n = p.hill_n[:, None]
    xn = xr**n
    hn = p.h[:, None] ** n
    denom = xn + hn
    # nan-safe division: when both numerator and denominator vanish the Hill term
    # is 0 (this only happens transiently during the init fixed point).
    safe_denom = jnp.where(denom > 0, denom, 1.0)
    hill = jnp.where(denom > 0, xn / safe_denom, 0.0)
    k = p.k[:, None]
    contrib = jnp.where(k > 0, k * hill, jnp.abs(k) * (1.0 - hill))
    contrib = contrib * p.edge_mask[:, None]
    return jax.ops.segment_sum(contrib, p.tar_idx, num_segments=g)


def _production(
    x: Array,
    p: SergioParams,
    prod_rates: Array,
    require_mrs: bool = True,
    regulated_basal_scale: float = 1.0,
) -> Array:
    """Total production rate per gene.

    Standard SERGIO (``require_mrs=True``): master regulators use their basal
    ``prod_rates`` and every other gene is driven purely by its Hill inputs -- which
    requires a DAG with master-regulator sources to have any drive.

    Permissive / cycle-tolerant mode (``require_mrs=False``): every gene gets its basal
    ``prod_rates`` plus its Hill inputs, so any GRN -- cyclic, with no source nodes -- is
    still driven. ``regulated_basal_scale`` scales the basal drive of *regulated*
    (non-master) genes: 1.0 keeps the fully-permissive basal-dominated regime, while a
    small value makes regulated genes regulation-dominated (so a knockout strongly and
    specifically shifts its targets) while master regulators keep full basal -- a middle
    ground between pure SERGIO and the permissive prior that stays cycle-tolerant.

    Args:
        x: Current concentration, ``(G, C)``.
        p: Network parameters.
        prod_rates: Basal production per gene and cell type, ``(G, C)``.
        require_mrs: Restrict basal production to master regulators.
        regulated_basal_scale: Basal multiplier for non-master genes (permissive mode).

    Returns:
        Non-negative production rate per gene and cell type, ``(G, C)``.
    """
    p_hill = _edge_production(x, p)
    if require_mrs:
        out = jnp.where(p.mr_mask[:, None] > 0, prod_rates, p_hill)
    else:
        basal_scale = jnp.where(p.mr_mask[:, None] > 0, 1.0, regulated_basal_scale)
        out = prod_rates * basal_scale + p_hill
    out = out * p.prod_scale[:, None]
    return jnp.maximum(out, 0.0)


def init_steady_state(p: SergioParams, cfg: SergioConfig) -> tuple[SergioParams, Array]:
    """Estimate per-edge half-responses and the steady-state concentration.

    Returns ``(params_with_h, ss)`` where ``ss`` has shape ``(G, C)``. Implemented
    as a fixed-point iteration that reproduces SERGIO's level-by-level estimate on
    a DAG: ``h_edge = mean_c(ss[reg])`` and ``ss = production(ss) / decay``.
    """
    g = p.num_genes
    c = cfg.num_cell_types
    n_iters = cfg.init_iters if cfg.init_iters is not None else g
    decay = p.decay[:, None]

    def body(carry, _):
        ss, _h = carry
        h_new = jnp.mean(ss[p.reg_idx], axis=1)  # (E,) mean over cell types of regulator ss
        p_h = dataclasses.replace(p, h=h_new)
        prod = _production(ss, p_h, p.prod_rates, cfg.require_mrs, cfg.regulated_basal_scale)
        ss_new = prod / decay
        return (ss_new, h_new), None

    ss0 = jnp.zeros((g, c), dtype=p.decay.dtype)
    (ss, h), _ = lax.scan(body, (ss0, p.h), None, length=max(n_iters, 1))
    return dataclasses.replace(p, h=h), ss


def simulate_trajectory(
    p: SergioParams, ss: Array, key: Array, cfg: SergioConfig, noise_s: Array | float | None = None
) -> Array:
    """Integrate the SERGIO SDE and return the full trajectory.

    Returns shape ``(max_iter + 1, G, C)`` including the steady-state column 0,
    matching SERGIO's ``sim_conc`` layout. The update (paper Eq. 3) is::

        x' = x + (p - lambda*x) dt
               + noise_s * sqrt(dt) * (sqrt(p) * eps_p + sqrt(lambda*x) * eps_d)

    ``noise_s`` overrides ``cfg.noise_s`` when given; pass a traced scalar to vary the
    SDE noise per simulation (``vmap``) without recompiling the static ``cfg``.
    """
    decay = p.decay[:, None]
    dt = cfg.dt
    sqrt_dt = jnp.sqrt(dt)
    ko = p.ko_mask[:, None]
    noise_s = cfg.noise_s if noise_s is None else noise_s

    def step(x, key_t):
        kp, kd = random.split(key_t)
        prod = _production(x, p, p.prod_rates, cfg.require_mrs, cfg.regulated_basal_scale)
        d = decay * x
        eps_p = random.normal(kp, x.shape)
        eps_d = random.normal(kd, x.shape)
        noise = (jnp.sqrt(prod) * eps_p + jnp.sqrt(d) * eps_d) * noise_s * sqrt_dt
        x_new = x + (prod - d) * dt + noise
        x_new = jnp.maximum(x_new, 0.0)
        x_new = x_new * (1.0 - ko)  # knocked-out genes stay silent
        return x_new, x_new

    keys = random.split(key, cfg.max_iter)
    _, traj = lax.scan(step, ss, keys)  # (max_iter, G, C)
    return jnp.concatenate([ss[None], traj], axis=0)  # prepend column 0


def sample_expression(traj: Array, key: Array, cfg: SergioConfig) -> Array:
    """Sample ``num_cells`` cells per cell type from the post-burn-in trajectory.

    Returns the expression matrix of shape ``(C * num_cells, G)`` (cells ordered
    by cell type, then cell), matching SERGIO's random time-step sampling.
    """
    t, g, c = traj.shape
    # Uniform indices in [safety_iter, max_iter), as in SERGIO's get_expr_df.
    idx = random.randint(key, (c, cfg.num_cells), cfg.safety_iter, cfg.max_iter)

    def gather_ct(ci):
        tc = traj[:, :, ci]  # (T, G)
        return tc[idx[ci]]  # (num_cells, G)

    expr = jax.vmap(gather_ct)(jnp.arange(c))  # (C, num_cells, G)
    return expr.reshape(c * cfg.num_cells, g)


def simulate(p: SergioParams, key: Array, cfg: SergioConfig, noise_s: Array | float | None = None) -> Array:
    """End-to-end clean (noise-free of technical effects) simulation.

    Runs steady-state init, SDE integration and cell sampling, returning the
    expression matrix of shape ``(C * num_cells, G)``. ``noise_s`` overrides
    ``cfg.noise_s`` when given (a traced scalar, e.g. sampled per network).
    """
    k_sim, k_sample = random.split(key)
    p_init, ss = init_steady_state(p, cfg)
    traj = simulate_trajectory(p_init, ss, k_sim, cfg, noise_s)
    return sample_expression(traj, k_sample, cfg)
