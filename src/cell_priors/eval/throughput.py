"""Throughput benchmark + advertising plot: how fast can we feed the model?

Compares four ways of delivering one SERGIO network's expression matrix
(``(num_cells, num_genes)``) to a training loop, as **networks per second**:

* ``sergio_rs``  -- the Rust reference simulator (single process);
* ``jax_cpu``    -- this library's JAX SERGIO core, vmapped, on CPU;
* ``jax_gpu``    -- the same JAX core, vmapped, on GPU;
* ``torch_h5``   -- load precomputed data from an ``.h5ad`` via a PyTorch
  ``DataLoader`` (the generate-to-disk-then-load pipeline).

The JAX variants run the *same* algorithm as ``sergio_rs`` (validated to match
numerically), so the only difference is the backend -- a fair comparison.

Usage (CPU and GPU need separate processes, since JAX fixes the platform at
startup)::

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= uv run python -m cell_priors.eval.throughput \
        measure --out throughput.json --extras
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python -m cell_priors.eval.throughput \
        measure --out throughput.json
    uv run python -m cell_priors.eval.throughput plot --data throughput.json \
        --out assets/throughput.png --device "Lenovo ThinkStation - NVIDIA GB10"
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import jax

from ..simulators.sergio import SergioConfig
from ..simulators.sergio.core import simulate as sergio_simulate
from ._common import build_sampler, matched_sergio_networks

CELLS = 128
BATCH = 64
REPEATS = 5
GENES = [50, 100, 200, 400]


def _best(fn, *args, repeats=REPEATS):
    jax.block_until_ready(fn(*args))  # compile / warm up
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out) if isinstance(out, jax.Array) else None
        best = min(best, time.perf_counter() - t)
    return best


def _bench_jax(ng: int) -> float:
    """Networks/sec for the JAX SERGIO core, vmapped over a batch."""
    cfg = SergioConfig(num_cells=CELLS, num_cell_types=1)
    params, _, _ = matched_sergio_networks(ng, 1, build_sampler(), cfg, seed=0)
    keys = jax.random.split(jax.random.PRNGKey(0), BATCH)
    fn = jax.jit(jax.vmap(lambda k: sergio_simulate(params, k, cfg)))
    return BATCH / _best(fn, keys)


def _bench_sergio_rs(ng: int) -> float:
    """Networks/sec for the Rust sergio_rs reference (single process)."""
    import sergio_rs

    cfg = SergioConfig(num_cells=CELLS, num_cell_types=1)
    _, grn, mr_profile = matched_sergio_networks(ng, 1, build_sampler(), cfg, seed=0)

    def run():
        sim = sergio_rs.Sim(
            grn,
            num_cells=CELLS,
            noise_s=cfg.noise_s,
            safety_iter=cfg.safety_iter,
            scale_iter=cfg.scale_iter,
            dt=cfg.dt,
            seed=0,
        )
        return sim.simulate(mr_profile)

    best = float("inf")
    for _ in range(REPEATS):
        t = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - t)
    return 1.0 / best


def _bench_torch_h5(ng: int, tmpdir: Path) -> float | None:
    """Networks/sec delivered from an .h5ad via a PyTorch DataLoader."""
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        return None

    import anndata as ad
    import numpy as np

    from ..io import generate_anndata
    from ..priors import MapPfnPrior

    # Pre-generate a dataset of networks and write it to disk.
    prior = MapPfnPrior(SergioConfig(num_cells=CELLS, num_cell_types=1), sampler=build_sampler())
    adata = generate_anndata(
        prior, jax.random.PRNGKey(0), num_contexts=4, num_genes=ng, treatments=list(range(8)), add_noise=True
    )
    path = tmpdir / f"net_{ng}.h5ad"
    adata.write_h5ad(path)

    adata = ad.read_h5ad(path)
    obs = adata.obs
    groups = obs.groupby(["context", "treatment"], observed=True).indices
    networks = [np.asarray(adata.X[idx], dtype=np.float32) for idx in groups.values()]

    class NetDataset(Dataset):
        def __len__(self):
            return len(networks)

        def __getitem__(self, i):
            return torch.from_numpy(networks[i])

    loader = DataLoader(NetDataset(), batch_size=1, num_workers=2, collate_fn=lambda b: b[0])
    # Time several passes over the loader.
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        n = 0
        for _batch in loader:
            n += 1
        dt = (time.perf_counter() - t) / max(n, 1)
        best = min(best, dt)
    return 1.0 / best


@click.group()
def main() -> None:
    """Throughput benchmark and plot."""


@main.command()
@click.option("--out", required=True, type=click.Path(), help="JSON results file (merged).")
@click.option("--extras/--no-extras", default=False, help="Also measure sergio_rs and torch_h5 (run on CPU).")
def measure(out, extras):
    """Measure variants available on the current JAX platform; merge into JSON."""
    import tempfile

    platform = jax.devices()[0].platform
    results = json.loads(Path(out).read_text()) if Path(out).exists() else {}
    results.setdefault("_meta", {})["cells"] = CELLS
    results["_meta"]["batch"] = BATCH
    results["_meta"]["genes"] = GENES

    jax_key = f"jax_{platform}"
    results.setdefault(jax_key, {})
    for ng in GENES:
        results[jax_key][str(ng)] = _bench_jax(ng)
        click.echo(f"{jax_key} genes={ng}: {results[jax_key][str(ng)]:.1f} nets/s")

    if extras:
        results.setdefault("sergio_rs", {})
        for ng in GENES:
            results["sergio_rs"][str(ng)] = _bench_sergio_rs(ng)
            click.echo(f"sergio_rs genes={ng}: {results['sergio_rs'][str(ng)]:.1f} nets/s")
        with tempfile.TemporaryDirectory() as td:
            torch_res = {}
            for ng in GENES:
                v = _bench_torch_h5(ng, Path(td))
                if v is not None:
                    torch_res[str(ng)] = v
                    click.echo(f"torch_h5 genes={ng}: {v:.1f} nets/s")
            if torch_res:
                results["torch_h5"] = torch_res

    Path(out).write_text(json.dumps(results, indent=2))
    click.echo(f"wrote {out}")


@main.command()
@click.option("--data", required=True, type=click.Path(exists=True))
@click.option("--out", default="assets/throughput.png", type=click.Path())
@click.option("--device", default="", help="Device label for the title.")
def plot(data, out, device):
    """Render the grouped-bar throughput figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = json.loads(Path(data).read_text())
    genes = results["_meta"]["genes"]
    series = [
        ("torch_h5", "h5 + PyTorch DataLoader (disk)", "#9aa0a6"),
        ("sergio_rs", "sergio_rs (Rust, CPU)", "#e8710a"),
        ("jax_cpu", "JAX SERGIO (CPU)", "#1a73e8"),
        ("jax_gpu", "JAX SERGIO (GPU)", "#34a853"),
    ]
    series = [(k, lbl, c) for k, lbl, c in series if k in results]

    x = np.arange(len(genes))
    n = len(series)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(9, 5.2))
    vmax = 0.0
    for i, (key, label, color) in enumerate(series):
        vals = [results[key].get(str(g), np.nan) for g in genes]
        vmax = max(vmax, np.nanmax(vals))
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            if v == v:  # not nan
                ax.annotate(f"{v:,.0f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=7.5)

    ax.set_yscale("log")
    ax.set_ylim(top=vmax * 2.2)  # headroom for value labels
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in genes])
    ax.set_xlabel("genes per network")
    ax.set_ylabel("networks / second  (higher is better, log scale)")
    title = "SERGIO prior throughput: same model, four backends"
    if device:
        title += f"\n{device}  ·  {results['_meta']['cells']} cells/net, batch {results['_meta']['batch']}"
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(series), frameon=False, fontsize=9)
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    click.echo(f"wrote {out}")


if __name__ == "__main__":
    main()
