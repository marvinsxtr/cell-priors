"""Interventions on a SERGIO GRN: hard knockouts and soft CRISPRi knockdowns.

Both are expressed as edits to :class:`SergioParams`, so an intervened network is
just another set of params with identical shapes -- it flows through the same
jitted simulation code.

**Hard knockout** removes the gene and its outgoing regulatory edges (matching
SERGIO's ``ko_perturbation``): the gene is silenced and stops regulating its
targets. Targets that lose their only regulator automatically become master
regulators (see :func:`recompute_mr_mask`).

**Soft knockdown** models CRISPRi: the gene's transcription is scaled down by a
factor ``strength`` in ``[0, 1]`` while the causal graph stays fully intact, so
the (attenuated) gene keeps regulating its targets. ``strength == 1`` recovers a
production-only knockout, but unlike the hard knockout the edges remain.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from jax import Array

from .grn import SergioParams, recompute_mr_mask


def _gene_onehot(gene_indices: Array, num_genes: int, dtype) -> Array:
    """Build a ``(num_genes,)`` {0,1} indicator for the targeted genes."""
    target = jnp.zeros(num_genes, dtype=dtype)
    return target.at[jnp.asarray(gene_indices).reshape(-1)].set(1.0)


def knockout(p: SergioParams, gene_indices: Array) -> SergioParams:
    """Hard knockout: silence the genes and remove their outgoing edges."""
    target = _gene_onehot(gene_indices, p.num_genes, p.decay.dtype)
    edge_mask = p.edge_mask * (1.0 - target[p.reg_idx])
    prod_scale = p.prod_scale * (1.0 - target)
    ko_mask = jnp.maximum(p.ko_mask, target)
    perturbed = dataclasses.replace(p, edge_mask=edge_mask, prod_scale=prod_scale, ko_mask=ko_mask)
    return recompute_mr_mask(perturbed)


def knockdown(p: SergioParams, gene_indices: Array, strength: float = 1.0) -> SergioParams:
    """Soft CRISPRi knockdown: scale production by ``(1 - strength)``, graph intact."""
    target = _gene_onehot(gene_indices, p.num_genes, p.decay.dtype)
    prod_scale = p.prod_scale * (1.0 - strength * target)
    return dataclasses.replace(p, prod_scale=prod_scale)
