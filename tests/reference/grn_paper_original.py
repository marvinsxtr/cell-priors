"""Original grn-paper RNA simulation, copied for numerical validation.

Adapted (only to take parameters directly instead of off an ``nx.DiGraph``) from
``src/grn.py`` of https://github.com/maguirre1/grn-paper -- the reference
implementation for Aguirre et al. 2025. The arithmetic of ``simulate_rna`` and
``observation_model`` is preserved verbatim. See ``LICENSE`` in this directory
(MIT, Copyright (c) 2023 maguirre1).
"""

from __future__ import annotations

import numpy as np
import scipy.special


def simulate_rna_reference(
    beta,
    alpha,
    l,
    x0=None,
    link=None,
    s=1e-4,
    dt=1e-2,
    tmax=20000,
    n=1,
    tol=1e-3,
    step=1000,
    burnin=5000,
):
    """Faithful copy of ``grn.simulate_rna`` operating on explicit parameters.

    ``beta`` is ``(G, G)`` with ``beta[i, j]`` the effect of regulator ``i`` on
    target ``j``; ``alpha`` and ``l`` are ``(G, 1)``. Returns ``(n, G)`` time-mean
    expression over the post-burn-in window (the reference observation model).
    """
    n_genes = len(alpha)
    if link is None:
        link = scipy.special.expit

    # traceline goes here (this could be made more efficient)
    X = np.zeros((n, tmax, n_genes))

    # set initial condition
    if x0 is None:
        X[:, 0, :] = np.zeros((n, n_genes))
    elif x0.shape == (n, n_genes):
        X[:, 0, :] = x0
    elif x0.shape == (n_genes,):
        X[:, 0, :] = np.vstack([x0 for _ in range(n)])

    # run simulations: x.shape=(tmax,self.n)
    for i in range(tmax - 1):
        dpos = link(alpha.T + X[:, i, :] @ beta)
        dneg = l.T * X[:, i, :]
        X[:, i + 1, :] = X[:, i, :] + dt * (dpos - dneg)
        X[:, i + 1, :] += s * np.sqrt(dt * X[:, i, :]) * np.random.normal(0, 1, size=(n, n_genes))
        X[:, i + 1, :] = np.maximum(0, X[:, i + 1, :])  # clip negative values

        # check convergence
        if i % step == 0 and i - step > burnin:
            now = X[0, burnin:i, :].mean(axis=0)
            then = X[0, burnin : (i - step), :].mean(axis=0)
            if np.max(np.abs(np.log2(now / then))[now > s]) < tol:
                X = X[:, :i, :]
                break

    # pass into observation model (mean over the post-burn-in window)
    return np.mean(X[:, burnin:i, :], axis=1)
