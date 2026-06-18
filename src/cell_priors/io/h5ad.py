"""Write prior-generated data to ``.h5ad`` in the MapPFN AnnData format.

The layout matches ``map_pfn.data`` so files are drop-in for MapPFN:

* ``adata.X``       -- expression counts, shape ``(num_cells, num_genes)``
* ``adata.var_names``-- gene names ``GENE0000``, ``GENE0001``, ...
* ``adata.obs['context']``   -- GRN/context id (one per sampled network)
* ``adata.obs['treatment']`` -- perturbed gene id as a string, or ``'control'``

For every context we emit a control condition plus one condition per treated
gene (matching SERGIO's control + per-gene-knockout design).
"""

from __future__ import annotations

import anndata as ad
import jax
import numpy as np
import pandas as pd
from jax import Array

from ..base import InterventionKind, Prior

# Column names / sentinel values mirrored from MapPFN (map_pfn.data.utils).
CONTEXT_COL = "context"
TREATMENT_COL = "treatment"
CONTROL_VALUE = "control"


def gene_names(num_genes: int) -> list[str]:
    """MapPFN-style gene identifiers."""
    return [f"GENE{i:04d}" for i in range(num_genes)]


def generate_anndata(
    prior: Prior,
    key: Array,
    num_contexts: int = 4,
    num_genes: int = 50,
    treatments: list[int] | None = None,
    kind: InterventionKind = InterventionKind.KNOCKOUT,
    strength: float = 1.0,
    add_noise: bool = True,
    noise_profile: str = "DS6",
    sample_kwargs: dict | None = None,
) -> ad.AnnData:
    """Generate a control + interventional dataset as an :class:`AnnData`.

    ``treatments`` is the list of gene indices to knock out per context (default:
    all genes). ``kind``/``strength`` select hard knockout vs soft knockdown.
    """
    sample_kwargs = sample_kwargs or {}
    treated = list(range(num_genes)) if treatments is None else list(treatments)
    var = pd.DataFrame(index=gene_names(num_genes))

    x_blocks: list[np.ndarray] = []
    contexts: list[str] = []
    treatment_ids: list[str] = []

    for ctx in range(num_contexts):
        ctx_key = jax.random.fold_in(key, ctx)
        k_params, k_ctrl = jax.random.split(ctx_key)
        params = prior.sample_params(k_params, num_genes=num_genes, **sample_kwargs)
        ctx_id = str(ctx)

        def _add(expr: Array, treatment_id: str, ctx_id: str = ctx_id) -> None:
            arr = np.asarray(expr)
            x_blocks.append(arr)
            contexts.extend([ctx_id] * arr.shape[0])
            treatment_ids.extend([treatment_id] * arr.shape[0])

        # Control condition.
        _add(prior.observational(params, k_ctrl, add_noise=add_noise, noise_profile=noise_profile), CONTROL_VALUE)

        # One interventional condition per treated gene.
        for gene in treated:
            t_key = jax.random.fold_in(ctx_key, gene + 1)
            expr = prior.interventional(
                params,
                t_key,
                np.array([gene]),
                kind=kind,
                strength=strength,
                add_noise=add_noise,
                noise_profile=noise_profile,
            )
            _add(expr, str(gene))

    x = np.concatenate(x_blocks, axis=0).astype(np.float32)
    obs = pd.DataFrame(
        {
            CONTEXT_COL: pd.Categorical(contexts),
            TREATMENT_COL: pd.Categorical(treatment_ids),
        }
    )
    obs.index = [f"cell_{i}" for i in range(x.shape[0])]
    return ad.AnnData(X=x, obs=obs, var=var)


def write_h5ad(adata: ad.AnnData, path: str) -> None:
    """Write an AnnData to disk as ``.h5ad``."""
    adata.write_h5ad(path)
