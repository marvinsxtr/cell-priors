"""Interventions on a BoolODE network: hard knockouts and soft CRISPRi knockdowns.

Both are edits to :class:`BoolodeParams`, so an intervened network is just another
set of params with identical shapes and flows through the same jitted core.

**Hard knockout** silences the gene (its mRNA and protein are forced to zero, so it
can no longer regulate) and removes its outgoing edges. A target that loses its
only activator this way is left with no active activator edge and therefore becomes
constitutively expressed automatically (the basal rule in :func:`core.activation`,
the analogue of SERGIO orphans becoming master regulators).

**Soft knockdown** models CRISPRi: the gene's transcription rate is scaled by
``1 - strength`` in ``[0, 1]`` while the graph stays fully intact, so the
(attenuated) gene keeps regulating its targets. ``strength == 1`` silences its
transcription but, unlike a knockout, the edges remain.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from jax import Array

from .core import BoolodeParams


def _gene_onehot(gene_indices: Array, num_genes: int, dtype) -> Array:
    target = jnp.zeros(num_genes, dtype=dtype)
    return target.at[jnp.asarray(gene_indices).reshape(-1)].set(1.0)


def knockout(prm: BoolodeParams, gene_indices: Array) -> BoolodeParams:
    """Hard knockout: silence the genes and remove their outgoing edges."""
    target = _gene_onehot(gene_indices, prm.num_genes, prm.m.dtype)
    edge_mask = prm.edge_mask * (1.0 - target[prm.reg_idx])
    prod_scale = prm.prod_scale * (1.0 - target)
    ko_mask = jnp.maximum(prm.ko_mask, target)
    return dataclasses.replace(prm, edge_mask=edge_mask, prod_scale=prod_scale, ko_mask=ko_mask)


def knockdown(prm: BoolodeParams, gene_indices: Array, strength: float = 1.0) -> BoolodeParams:
    """Soft CRISPRi knockdown: scale transcription by ``(1 - strength)``, graph intact."""
    target = _gene_onehot(gene_indices, prm.num_genes, prm.m.dtype)
    prod_scale = prm.prod_scale * (1.0 - strength * target)
    return dataclasses.replace(prm, prod_scale=prod_scale)
