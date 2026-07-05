"""Cross-simulator throughput benchmark: SERGIO vs. grn-paper vs. BoolODE.

Measures how fast each JAX simulator generates whole networks **end to end** --
sampling the GRN structure *and* kinetics *and* simulating expression, batched in
one ``jit``/``vmap`` with no host round-trip -- as **networks per second** across
gene counts. All three priors share the grouped scale-free sampler and differ only
in the expression model, so this isolates the cost of each simulator.

Renders the same grouped-bar figure as the README throughput plot and writes the
identical numbers as a Markdown table (for ``assets/``). Usage (CPU and GPU need
separate processes since JAX fixes the platform at startup)::

    uv run python -m cell_priors.eval.benchmark_simulators measure --out assets/simulator_throughput.json
    uv run python -m cell_priors.eval.benchmark_simulators report \
        --data assets/simulator_throughput.json --device "my machine"
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import jax

from ._common import build_prior

CELLS = 64
BATCH = 8
REPEATS = 4
GENES = [20, 50, 100]

# (key, label, build_prior kwargs). "mappfn" is the pure-JAX cycle-tolerant SERGIO.
SIMULATORS = [
    ("mappfn", "SERGIO (Hill SDE)", dict(num_cell_types=1)),
    ("grn_paper", "grn-paper (sigmoid SDE)", dict()),
    ("boolode", "BoolODE (mRNA+protein)", dict()),
]
COLORS = {"mappfn": "#1a73e8", "grn_paper": "#a142f4", "boolode": "#34a853"}


def _best(fn, *args, repeats=REPEATS):
    jax.block_until_ready(fn(*args))  # compile / warm up
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t)
    return best


def _bench(simulator: str, ng: int, cfg_kwargs: dict) -> float:
    """Networks/sec for one simulator's end-to-end JAX prior, vmapped over a batch."""
    prior = build_prior(simulator, num_cells=CELLS, **cfg_kwargs)

    def one(k):
        k_struct, k_sim = jax.random.split(k)
        params = prior.sample_params(k_struct, num_genes=ng)
        return prior.observational(params, k_sim)

    keys = jax.random.split(jax.random.PRNGKey(0), BATCH)
    fn = jax.jit(jax.vmap(one))
    return BATCH / _best(fn, keys)


@click.group()
def main() -> None:
    """Cross-simulator throughput benchmark and report."""


@main.command()
@click.option("--out", required=True, type=click.Path(), help="JSON results file (merged).")
def measure(out):
    """Measure every simulator's end-to-end throughput on the current JAX platform."""
    platform = jax.devices()[0].platform
    results = json.loads(Path(out).read_text()) if Path(out).exists() else {}
    meta = results.setdefault("_meta", {})
    meta.update(cells=CELLS, batch=BATCH, genes=GENES)

    for sim, label, cfg_kwargs in SIMULATORS:
        key = f"{sim}_{platform}"
        results.setdefault(key, {})
        for ng in GENES:
            v = _bench(sim, ng, cfg_kwargs)
            results[key][str(ng)] = v
            click.echo(f"{label:>28} [{platform}] genes={ng}: {v:.1f} nets/s")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2))
    click.echo(f"wrote {out}")


def _series(results):
    """Available ``(sim, label, platform, values-by-gene)`` series in a results dict."""
    genes = results["_meta"]["genes"]
    for sim, label, _ in SIMULATORS:
        for platform in ("gpu", "cpu"):
            key = f"{sim}_{platform}"
            if key in results:
                suffix = "" if len([p for p in ("gpu", "cpu") if f"{sim}_{p}" in results]) == 1 else f" [{platform}]"
                yield sim, label + suffix, platform, [results[key].get(str(g)) for g in genes]


def _write_table(results, path):
    genes = results["_meta"]["genes"]
    meta = results["_meta"]
    lines = [
        "# Simulator throughput (networks / second)",
        "",
        f"End-to-end JAX priors (sample GRN + kinetics + simulate), vmapped batch of "
        f"{meta['batch']}, {meta['cells']} cells/network. Higher is better.",
        "",
        "| simulator | platform | " + " | ".join(f"{g} genes" for g in genes) + " |",
        "|---|---|" + "|".join("---" for _ in genes) + "|",
    ]
    for _sim, label, platform, vals in _series(results):
        cells = " | ".join("—" if v is None else f"{v:,.0f}" for v in vals)
        lines.append(f"| {label} | {platform} | {cells} |")
    Path(path).write_text("\n".join(lines) + "\n")
    return path


def _plot(results, path, device):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    genes = results["_meta"]["genes"]
    series = list(_series(results))
    x = np.arange(len(genes))
    n = len(series)
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    vmax = 0.0
    for i, (sim, label, platform, vals) in enumerate(series):
        vals = [np.nan if v is None else v for v in vals]
        vmax = max(vmax, np.nanmax(vals))
        hatch = "//" if platform == "cpu" else None
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=label, color=COLORS[sim], hatch=hatch)
        for b, v in zip(bars, vals):
            if v == v:
                ax.annotate(f"{v:,.0f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=7.5)

    ax.set_yscale("log")
    ax.set_ylim(top=vmax * 2.2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in genes])
    ax.set_xlabel("genes per network")
    ax.set_ylabel("networks / second  (higher is better, log scale)")
    title = "Prior throughput by simulator: on-device end-to-end generation"
    if device:
        title += f"\n{device}  ·  {results['_meta']['cells']} cells/net, batch {results['_meta']['batch']}"
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(n, 3), frameon=False, fontsize=9)
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)


@main.command()
@click.option("--data", required=True, type=click.Path(exists=True))
@click.option("--fig", default="assets/simulator_throughput.png", type=click.Path())
@click.option("--table", default="assets/simulator_throughput.md", type=click.Path())
@click.option("--device", default="", help="Device label for the title.")
def report(data, fig, table, device):
    """Write the grouped-bar figure and the Markdown table from measured results."""
    results = json.loads(Path(data).read_text())
    _plot(results, fig, device)
    _write_table(results, table)
    click.echo(f"wrote {fig} and {table}")


if __name__ == "__main__":
    main()
