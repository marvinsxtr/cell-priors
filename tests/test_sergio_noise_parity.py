"""Numerical parity of the JAX SERGIO core against sergio_rs with SDE noise (``noise_s > 0``).

``test_sergio_parity`` only checks the deterministic steady state (``noise_s = 0``), so the
stochastic dynamics were never validated. This checks the noise-driven population distributions:
at a large sample size the per-gene population means, for control and for a knockout, match
sergio_rs. (Individual trajectories differ because the RNGs differ, so we compare the population
means via correlation and regression slope rather than exact equality.)
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import sergio_rs
from conftest import build_matched_grn, gene_name

from cell_priors.simulators.sergio import core
from cell_priors.simulators.sergio.interventions import knockout
from cell_priors.simulators.sergio.params import SergioConfig, make_params

DT, SAFETY, SCALE, NCELL = 0.01, 400, 5, 2000


def _sergio_pop_means(grn, mr_profile, num_genes, noise_s):
    df = sergio_rs.Sim(
        grn, num_cells=NCELL, noise_s=noise_s, safety_iter=SAFETY, scale_iter=SCALE, dt=DT, seed=1
    ).simulate(mr_profile)
    names = df["Genes"].to_list()
    data = df.drop("Genes").to_numpy()
    out = np.zeros((num_genes, data.shape[1]))
    for row, nm in zip(data, names):
        out[int(nm[4:])] = row
    return out.mean(1)  # (num_genes,) per-gene population mean


def _matched():
    # A fixed multi-level DAG (integer Hill, all activating) with several master regulators.
    edges = [
        (0, 2),
        (1, 2),
        (2, 4),
        (3, 4),
        (2, 5),
        (4, 6),
        (5, 6),
        (0, 7),
        (7, 8),
        (6, 9),
        (8, 9),
        (4, 10),
        (9, 11),
        (10, 11),
    ]
    ng = 12
    rng = np.random.default_rng(0)
    decay = rng.uniform(0.5, 1.0, ng)
    hill = np.full(len(edges), 2)
    k = rng.uniform(1.0, 5.0, len(edges))
    grn, mrp, num_genes, reg, tar, karr, harr, dec = build_matched_grn(edges, decay, hill, k, 1, 7)
    return grn, mrp, num_genes, reg, tar, karr, harr, dec


def _params(reg, tar, karr, harr, dec, basal_ss):
    return make_params(reg, tar, karr, harr, dec, basal_ss[:, None] * dec[:, None])


def _cfg(num_genes):
    return SergioConfig(
        num_cells=NCELL,
        num_cell_types=1,
        safety_iter=SAFETY,
        scale_iter=SCALE,
        dt=DT,
        noise_s=1.0,
        require_mrs=True,
        init_iters=num_genes,
    )


def test_control_population_parity_with_noise():
    grn, mrp, num_genes, reg, tar, karr, harr, dec = _matched()
    s0 = _sergio_pop_means(grn, mrp, num_genes, 0.0)  # deterministic steady state -> MR basals
    p = _params(reg, tar, karr, harr, dec, s0)
    sm = _sergio_pop_means(grn, mrp, num_genes, 1.0)
    jm = np.asarray(core.simulate(p, jax.random.PRNGKey(0), _cfg(num_genes), 1.0)).mean(0)
    corr = np.corrcoef(sm, jm)[0, 1]
    slope = float(np.polyfit(sm, jm, 1)[0])
    assert corr > 0.97, f"control population means diverge from sergio_rs (corr={corr:.3f})"
    assert 0.9 < slope < 1.1, f"control population slope off (slope={slope:.3f})"


def test_knockout_population_parity_with_noise():
    grn, mrp, num_genes, reg, tar, karr, harr, dec = _matched()
    s0 = _sergio_pop_means(grn, mrp, num_genes, 0.0)
    p = _params(reg, tar, karr, harr, dec, s0)
    gko = int(reg[0])  # a regulator gene
    # sergio_rs post-knockout: fresh grn (ko_perturbation mutates the shared graph via set_mrs)
    grn_ko, mrp_ko, num_genes, *_ = _matched()
    grn_ko, mrp_ko = grn_ko.ko_perturbation(gene_name=gene_name(gko), mr_profile=mrp_ko)
    sko0 = _sergio_pop_means(grn_ko, mrp_ko, num_genes, 0.0)  # post-KO steady state -> post-KO basals
    sm = _sergio_pop_means(grn_ko, mrp_ko, num_genes, 1.0)
    p_ko = knockout(p, jnp.array([gko]))
    p_ko = dataclasses.replace(p_ko, prod_rates=jnp.asarray(sko0[:, None] * dec[:, None]))
    jm = np.asarray(core.simulate(p_ko, jax.random.PRNGKey(0), _cfg(num_genes), 1.0)).mean(0)
    corr = np.corrcoef(sm, jm)[0, 1]
    slope = float(np.polyfit(sm, jm, 1)[0])
    assert corr > 0.97, f"knockout population means diverge from sergio_rs (corr={corr:.3f})"
    assert 0.9 < slope < 1.1, f"knockout population slope off (slope={slope:.3f})"
