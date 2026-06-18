"""Benchmark prior simulation speed across dimensionalities and implementations.

Compares the JAX simulators (compiled, single network and vmapped batch) against
the ``sergio_rs`` Rust reference where applicable, at several ``num_genes`` sizes.
All networks come from the grouped scale-free sampler. Clean CLI, no notebooks::

    uv run python -m cell_priors.eval.benchmark_speed --genes 50,100,200 --cells 200
    uv run python -m cell_priors.eval.benchmark_speed --simulator grn_paper --batch 8
"""

from __future__ import annotations

import click
import jax

from ..simulators.sergio import SergioConfig
from ..simulators.sergio.core import simulate as sergio_simulate
from ._common import build_prior, build_sampler, matched_sergio_networks, timeit


def _bench_jax(sim_fn, params, batch, repeats):
    key = jax.random.PRNGKey(0)
    if batch > 1:
        keys = jax.random.split(key, batch)
        fn = jax.jit(jax.vmap(lambda k: sim_fn(params, k)))
        jax.block_until_ready(fn(keys))  # compile
        return timeit(lambda: fn(keys), repeats)
    fn = jax.jit(sim_fn)
    jax.block_until_ready(fn(params, key))  # compile
    return timeit(lambda: fn(params, key), repeats)


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
@click.option("--simulator", default="sergio", type=click.Choice(["sergio", "mappfn", "grn_paper"]))
@click.option("--genes", default="20,50,100", help="Comma-separated gene counts.")
@click.option("--cells", default=200, help="Cells per cell type (sergio) / cells (grn_paper).")
@click.option("--cell-types", default=1, help="Number of cell types (sergio only).")
@click.option("--batch", default=1, help="vmap batch size for the JAX prior.")
@click.option("--repeats", default=3)
@click.option("--rust/--no-rust", default=True, help="Also benchmark sergio_rs (sergio only).")
@click.option("--seed", default=0)
def main(simulator, genes, cells, cell_types, batch, repeats, rust, seed):
    """Run the speed benchmark and print a comparison table."""
    gene_list = [int(g) for g in genes.split(",")]
    sampler = build_sampler()
    use_rust = rust and simulator == "sergio"
    print(f"device={jax.devices()[0].platform}  simulator={simulator}  cells={cells}  batch={batch}")
    header = f"{'genes':>6} {'edges':>6} {'jax(ms)':>10} {'jax/net(ms)':>12}"
    if use_rust:
        header += f" {'rust(ms)':>10} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for ng in gene_list:
        if simulator == "sergio":
            cfg = SergioConfig(num_cells=cells, num_cell_types=cell_types)
            params, grn, mr_profile = matched_sergio_networks(ng, cell_types, sampler, cfg, seed)
            sim_fn = lambda p, k, cfg=cfg: sergio_simulate(p, k, cfg)  # noqa: E731
            edges = int(params.num_edges)
        else:
            prior = build_prior(simulator, num_cells=cells)
            params = prior.sample_params(jax.random.PRNGKey(seed), num_genes=ng)
            sim_fn = lambda p, k, prior=prior: prior.observational(p, k)  # noqa: E731
            edges = int(getattr(params, "num_edges", 0))

        t_jax = _bench_jax(sim_fn, params, batch, repeats)
        per_net = t_jax / batch
        row = f"{ng:>6} {edges:>6} {t_jax * 1e3:>10.2f} {per_net * 1e3:>12.2f}"
        if use_rust:
            t_rust = _bench_rust(grn, mr_profile, cfg, repeats) * batch
            row += f" {t_rust * 1e3:>10.2f} {t_rust / t_jax:>8.2f}x"
        print(row)


if __name__ == "__main__":
    main()
