"""Compare priors to each other and to real datasets.

A single clean CLI (no notebooks) for the everyday questions when developing a
prior: is the output healthy, what are its distributional properties, which
genes are differentially expressed under a perturbation, and how close is it to a
real dataset.

A *source* is one of:

* ``sergio``                          -- generate fresh data from the SERGIO prior
* ``path/to/file.h5ad``               -- a local AnnData file
* ``hf:marvinsxtr/MapPFN/frangieh.h5ad`` -- a file in a Hugging Face dataset repo

Examples::

    uv run python -m cell_priors.eval.compare stats sergio
    uv run python -m cell_priors.eval.compare de sergio --gene 0
    uv run python -m cell_priors.eval.compare distribution sergio hf:marvinsxtr/MapPFN/frangieh.h5ad
"""

from __future__ import annotations

import click
import numpy as np

from ..base import InterventionKind
from ..io import generate_anndata
from ..utils import summarize
from ._common import build_prior

# Source names that generate fresh data from a prior (sampler x simulator).
PRIOR_SOURCES = {
    "sergio": dict(num_cell_types=1, safety_iter=120, scale_iter=5),
    "mappfn": dict(num_cell_types=1, safety_iter=120, scale_iter=5),
    "grn_paper": dict(),
    "boolode": dict(),
}


def load_source(
    source: str,
    genes: int = 50,
    cells: int = 100,
    cell_types: int = 1,
    contexts: int = 3,
    kind: str = "knockout",
    strength: float = 1.0,
    seed: int = 0,
):
    """Resolve a source string to an :class:`AnnData`.

    ``sergio`` / ``grn_paper`` generate fresh data; ``path.h5ad`` reads a local
    file; ``hf:<repo>/<file>`` downloads from a Hugging Face dataset.
    """
    import anndata as ad
    import jax

    if source in PRIOR_SOURCES:
        cfg = dict(PRIOR_SOURCES[source])
        cfg["num_cells"] = cells
        prior = build_prior(source, **cfg)
        return generate_anndata(
            prior,
            jax.random.PRNGKey(seed),
            num_contexts=contexts,
            num_genes=genes,
            kind=InterventionKind(kind),
            strength=strength,
            add_noise=True,
        )
    if source.startswith("hf:"):
        from huggingface_hub import hf_hub_download

        rest = source[3:]
        repo_id, filename = rest.rsplit("/", 1)
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        return ad.read_h5ad(path)
    return ad.read_h5ad(source)


def _gene_summary(x: np.ndarray) -> dict[str, np.ndarray]:
    lx = np.log1p(x)
    return {
        "gene_mean": lx.mean(axis=0),
        "gene_var": lx.var(axis=0),
        "dropout": (x == 0).mean(axis=0),
        "library_size": x.sum(axis=1),
    }


@click.group()
def main() -> None:
    """Prior comparison and debugging tools."""


@main.command()
@click.argument("source")
@click.option("--genes", default=50)
@click.option("--cells", default=100)
@click.option("--cell-types", default=1)
def stats(source, genes, cells, cell_types):
    """Print health and distributional statistics for a source."""
    adata = load_source(source, genes=genes, cells=cells, cell_types=cell_types)
    x = np.asarray(adata.X)
    s = summarize(x)
    print(f"source: {source}   shape: {x.shape}")
    for key, val in s.as_dict().items():
        print(f"  {key:>18}: {val}")
    if "treatment" in adata.obs:
        print(f"  {'n_treatments':>18}: {adata.obs['treatment'].nunique()}")
        print(f"  {'n_contexts':>18}: {adata.obs.get('context', []).nunique() if 'context' in adata.obs else 'n/a'}")


@main.command()
@click.argument("source")
@click.option("--gene", default=0, help="Perturbed gene id to test (treatment value).")
@click.option("--genes", default=50)
@click.option("--cells", default=200)
@click.option("--top", default=10, help="Number of top DE genes to print.")
def de(source, gene, genes, cells, top):
    """Differential-expression genes: a perturbation vs control (scanpy)."""
    import scanpy as sc

    adata = load_source(source, genes=genes, cells=cells)
    if "treatment" not in adata.obs:
        raise click.ClickException("Source has no 'treatment' column; cannot run DE.")
    treatment = str(gene)
    groups = set(adata.obs["treatment"].astype(str))
    if treatment not in groups or "control" not in groups:
        raise click.ClickException(f"Need 'control' and '{treatment}' in treatments; have {sorted(groups)[:8]}...")

    sub = adata[adata.obs["treatment"].astype(str).isin([treatment, "control"])].copy()
    sub.X = sub.X.astype(float)
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    sub.obs["treatment"] = sub.obs["treatment"].astype(str).astype("category")
    sc.tl.rank_genes_groups(sub, groupby="treatment", groups=[treatment], reference="control", method="wilcoxon")
    names = sub.uns["rank_genes_groups"]["names"][treatment][:top]
    lfc = sub.uns["rank_genes_groups"]["logfoldchanges"][treatment][:top]
    pvals = sub.uns["rank_genes_groups"]["pvals_adj"][treatment][:top]
    print(f"Top {top} DE genes for treatment={treatment} vs control:")
    print(f"  {'gene':>10} {'log2FC':>10} {'padj':>12}")
    for name_, lfc_, p_ in zip(names, lfc, pvals):
        print(f"  {name_:>10} {lfc_:>10.3f} {p_:>12.3e}")


@main.command()
@click.argument("source_a")
@click.argument("source_b")
@click.option("--genes", default=50)
@click.option("--cells", default=200)
@click.option("--out", default="comparison.png", help="Output figure path.")
def distribution(source_a, source_b, genes, cells, out):
    """Compare the distributional properties of two sources (with a figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp, wasserstein_distance

    xa = np.asarray(load_source(source_a, genes=genes, cells=cells).X, dtype=float)
    xb = np.asarray(load_source(source_b, genes=genes, cells=cells).X, dtype=float)
    sa, sb = _gene_summary(xa), _gene_summary(xb)

    print(f"A = {source_a}  {xa.shape}")
    print(f"B = {source_b}  {xb.shape}")
    print(f"  {'metric':>14} {'KS stat':>10} {'wasserstein':>12}")
    for metric in ("gene_mean", "gene_var", "dropout", "library_size"):
        ks = ks_2samp(sa[metric], sb[metric]).statistic
        wd = wasserstein_distance(sa[metric], sb[metric])
        print(f"  {metric:>14} {ks:>10.4f} {wd:>12.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, metric in zip(axes.ravel(), ("gene_mean", "gene_var", "dropout", "library_size")):
        ax.hist(sa[metric], bins=30, alpha=0.5, density=True, label="A")
        ax.hist(sb[metric], bins=30, alpha=0.5, density=True, label="B")
        ax.set_title(metric)
        ax.legend()
    fig.suptitle(f"A={source_a}  vs  B={source_b}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    main()
