"""Reference BoolODE ODE model for numerical validation.

An *independent* implementation of the published BoolODE model (Pratapa et al.,
Nat. Methods 2020) -- **not** a copy of the BoolODE source (which is GPL-3.0). It
builds the transcriptional activation function the way BoolODE documents it: by
enumerating every combination of a gene's regulators, evaluating the gene's
Boolean rule on that combination to get the 0/1 coefficient, and summing the
corresponding products of Hill terms::

    f_g = ( alpha_0 + sum_C a_C * prod_{r in C} (p_r/k_r)^n_r )
          / ( 1      + sum_C       prod_{r in C} (p_r/k_r)^n_r )

with the standard signed-network rule

    gene ON  <=>  (any activator present) AND NOT (any repressor present),

extended so a gene with no activators is constitutively expressed (alpha_0 = 1,
repressible). The mRNA/protein ODEs and integration match BoolODE::

    dx_g/dt = m_g * f_g(p) - l_x_g * x_g,   dp_g/dt = r_g * x_g - l_p_g * p_g.

The combinatorial ``2^indeg`` expansion here is exactly what the JAX core computes
in ``O(E)`` closed form, so agreement validates that reimplementation.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.integrate import odeint


def _gene_regulators(num_genes, reg_idx, tar_idx, is_act, thr, hill_n):
    """Group active edges by target gene into per-gene regulator descriptions."""
    regs = {g: [] for g in range(num_genes)}
    for e in range(len(reg_idx)):
        regs[int(tar_idx[e])].append((int(reg_idx[e]), bool(is_act[e]), float(thr[e]), float(hill_n[e])))
    return regs


def _activation_fn(regs):
    """Build ``f_g(p)`` for one gene from its regulator list (closure over combos)."""
    has_act = any(a for _, a, _, _ in regs)
    n_reg = len(regs)

    def rule_on(on_mask):
        # on_mask: which of this gene's regulators are ON.
        any_act = any(regs[i][1] and on_mask[i] for i in range(n_reg))
        any_rep = any((not regs[i][1]) and on_mask[i] for i in range(n_reg))
        if has_act:
            return int(any_act and not any_rep)
        return int(not any_rep)  # no activators -> constitutive, repressible

    alpha_0 = rule_on([False] * n_reg)

    def f(p):
        if n_reg == 0:
            return float(alpha_0)  # source: constitutively 1
        # Concentrations are non-negative; guard the integrator's transient so a
        # negative base never reaches a fractional power (the fixed point is unchanged).
        hills = [(max(p[reg], 0.0) / thr) ** n for reg, _, thr, n in regs]
        num = float(alpha_0)
        den = 1.0
        for i in range(1, n_reg + 1):
            for combo in combinations(range(n_reg), i):
                prod = 1.0
                for j in combo:
                    prod *= hills[j]
                on = [False] * n_reg
                for j in combo:
                    on[j] = True
                den += prod
                num += rule_on(on) * prod
        return num / den

    return f


def boolode_steady_state(
    num_genes,
    reg_idx,
    tar_idx,
    is_act,
    thr,
    hill_n,
    m,
    l_x,
    r,
    l_p,
    tmax=400.0,
    n_points=4000,
):
    """Integrate the deterministic BoolODE ODE to steady state; return mRNA ``x`` (G,)."""
    regs = _gene_regulators(num_genes, reg_idx, tar_idx, is_act, thr, hill_n)
    fns = [_activation_fn(regs[g]) for g in range(num_genes)]
    m, l_x, r, l_p = (np.asarray(v, dtype=float) for v in (m, l_x, r, l_p))

    def model(Y, t):
        x = Y[:num_genes]
        p = Y[num_genes:]
        f = np.array([fns[g](p) for g in range(num_genes)])
        dx = m * f - l_x * x
        dp = r * x - l_p * p
        return np.concatenate([dx, dp])

    x0 = np.ones(num_genes)
    p0 = (r / l_p) * x0
    tspan = np.linspace(0.0, tmax, n_points)
    sol = odeint(model, np.concatenate([x0, p0]), tspan)
    return sol[-1, :num_genes]
