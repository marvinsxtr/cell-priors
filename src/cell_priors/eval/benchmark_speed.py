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
from ._common import build_prior, matched_sergio_networks, timeit


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


def _bench_jax_end_to_end(prior, num_genes, batch, repeats):
    """Time the full prior -- structure sampling + kinetics + simulation -- in one jit.

    For a batch this vmaps over keys, so each network in the batch is sampled and
    simulated independently (distinct structures), the regime that matters for
    on-device pretraining. Requires a prior whose parameter build is pure JAX.
    """
    key = jax.random.PRNGKey(0)

    def one(k):
        k_struct, k_sim = jax.random.split(k)
        params = prior.sample_params(k_struct, num_genes=num_genes)
        return prior.observational(params, k_sim)

    if batch > 1:
        keys = jax.random.split(key, batch)
        fn = jax.jit(jax.vmap(one))
        jax.block_until_ready(fn(keys))  # compile
        return timeit(lambda: fn(keys), repeats)
    fn = jax.jit(one)
    jax.block_until_ready(fn(key))  # compile
    return timeit(lambda: fn(key), repeats)


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
    use_rust = rust and simulator == "sergio"
    # The grn-paper prior builds its kinetics in pure JAX, so the whole prior (structure
    # sampling + simulation) is timed end to end; the SERGIO kinetics adapter is host-side,
    # so those JAX numbers are simulation only.
    end_to_end = simulator == "grn_paper"
    jax_mode = "sample+simulate (end to end)" if end_to_end else "simulate only"
    print(f"device={jax.devices()[0].platform}  simulator={simulator}  cells={cells}  batch={batch}  jax={jax_mode}")
    header = f"{'genes':>6} {'edges':>6} {'jax(ms)':>10} {'jax/net(ms)':>12}"
    if use_rust:
        header += f" {'rust(ms)':>10} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for ng in gene_list:
        if simulator == "sergio":
            cfg = SergioConfig(num_cells=cells, num_cell_types=cell_types)
            params, grn, mr_profile = matched_sergio_networks(ng, cell_types, cfg, seed)
            sim_fn = lambda p, k, cfg=cfg: sergio_simulate(p, k, cfg)  # noqa: E731
            edges = int(params.num_edges)
            t_jax = _bench_jax(sim_fn, params, batch, repeats)
        else:
            prior = build_prior(simulator, num_cells=cells)
            params = prior.sample_params(jax.random.PRNGKey(seed), num_genes=ng)
            edges = int((jax.numpy.asarray(params.beta) != 0).sum()) if end_to_end else int(params.num_edges)
            if end_to_end:
                t_jax = _bench_jax_end_to_end(prior, ng, batch, repeats)
            else:
                sim_fn = lambda p, k, prior=prior: prior.observational(p, k)  # noqa: E731
                t_jax = _bench_jax(sim_fn, params, batch, repeats)

        per_net = t_jax / batch
        row = f"{ng:>6} {edges:>6} {t_jax * 1e3:>10.2f} {per_net * 1e3:>12.2f}"
        if use_rust:
            t_rust = _bench_rust(grn, mr_profile, cfg, repeats) * batch
            row += f" {t_rust * 1e3:>10.2f} {t_rust / t_jax:>8.2f}x"
        print(row)


if __name__ == "__main__":
    main()
