"""Benchmark prior simulation speed across dimensionalities and implementations.

Compares the JAX SERGIO prior (compiled, single network and vmapped batch)
against the ``sergio_rs`` Rust reference at several ``(num_genes, num_cells)``
sizes. Clean CLI, no notebooks::

    uv run python -m cell_priors.eval.benchmark_speed --genes 50,100,200 --cells 200
    uv run python -m cell_priors.eval.benchmark_speed --batch 8 --no-rust
"""

from __future__ import annotations

import click
import jax
import numpy as np

from ..priors.sergio import SergioConfig
from ..priors.sergio.core import simulate
from ._common import build_matched_networks, timeit


def _bench_jax(params, cfg, batch, repeats):
    sim = jax.jit(lambda p, k: simulate(p, k, cfg))
    key = jax.random.PRNGKey(0)
    if batch > 1:
        keys = jax.random.split(key, batch)
        fn = jax.jit(jax.vmap(lambda k: simulate(params, k, cfg)))
        # compile
        jax.block_until_ready(fn(keys))
        return timeit(lambda: fn(keys), repeats)
    jax.block_until_ready(sim(params, key))  # compile
    return timeit(lambda: sim(params, key), repeats)


def _bench_rust(grn, mr_profile, cfg, repeats):
    import sergio_rs

    def run():
        sim = sergio_rs.Sim(
            grn,
            num_cells=cfg.num_cells,
            noise_s=cfg.noise_s,
            safety_iter=cfg.safety_iter,
            scale_iter=cfg.scale_iter,
            dt=cfg.dt,
            seed=0,
        )
        return sim.simulate(mr_profile)

    return timeit(run, repeats)


@click.command()
@click.option("--genes", default="20,50,100", help="Comma-separated gene counts.")
@click.option("--cells", default=200, help="Cells per cell type.")
@click.option("--cell-types", default=1, help="Number of cell types.")
@click.option("--avg-regulators", default=2.0, help="Average regulators per gene.")
@click.option("--batch", default=1, help="vmap batch size for the JAX prior.")
@click.option("--safety-iter", default=150)
@click.option("--scale-iter", default=10)
@click.option("--repeats", default=3)
@click.option("--rust/--no-rust", default=True, help="Also benchmark sergio_rs.")
@click.option("--seed", default=0)
def main(genes, cells, cell_types, avg_regulators, batch, safety_iter, scale_iter, repeats, rust, seed):
    """Run the speed benchmark and print a comparison table."""
    gene_list = [int(g) for g in genes.split(",")]
    print(f"device={jax.devices()[0].platform}  cells={cells}  cell_types={cell_types}  batch={batch}")
    header = f"{'genes':>6} {'edges':>6} {'jax(ms)':>10} {'jax/net(ms)':>12}"
    if rust:
        header += f" {'rust(ms)':>10} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for ng in gene_list:
        cfg = SergioConfig(
            num_cells=cells,
            num_cell_types=cell_types,
            safety_iter=safety_iter,
            scale_iter=scale_iter,
        )
        params, grn, mr_profile = build_matched_networks(ng, cell_types, avg_regulators, seed)
        t_jax = _bench_jax(params, cfg, batch, repeats)
        per_net = t_jax / batch
        row = f"{ng:>6} {params.num_edges:>6} {t_jax * 1e3:>10.2f} {per_net * 1e3:>12.2f}"
        if rust:
            t_rust = _bench_rust(grn, mr_profile, cfg, repeats) * batch  # rust does `batch` networks serially
            row += f" {t_rust * 1e3:>10.2f} {t_rust / t_jax:>8.2f}x"
        print(row)


if __name__ == "__main__":
    main()
