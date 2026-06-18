"""Uniform interface shared by all priors.

A *prior* generates synthetic single-cell expression from some latent generative
process (e.g. SERGIO). The contract is deliberately small so different priors are
interchangeable in benchmarks and comparisons:

* :meth:`sample_params` -- draw a generative configuration (a JAX pytree).
* :meth:`observational` -- simulate control expression ``(num_cells, num_genes)``.
* :meth:`intervene` -- return new params with one or more genes perturbed.
* :meth:`interventional` -- intervene, then simulate.

Params are pytrees of arrays and all sampling methods are pure functions of
``(params, key)``, so a prior composes with a JAX model inside a single jitted
graph with no host round-trip.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from jax import Array


class InterventionKind(StrEnum):
    """Supported intervention semantics."""

    KNOCKOUT = "knockout"  # hard: gene and its regulatory edges removed
    KNOCKDOWN = "knockdown"  # soft (CRISPRi): production scaled, graph intact


class Prior(ABC):
    """Abstract base class for generative priors over single-cell expression."""

    @abstractmethod
    def sample_params(self, key: Array, **kwargs: Any) -> Any:
        """Sample a generative configuration (a pytree of arrays)."""

    @abstractmethod
    def observational(self, params: Any, key: Array, **kwargs: Any) -> Array:
        """Simulate observational expression, shape ``(num_cells, num_genes)``."""

    @abstractmethod
    def intervene(
        self,
        params: Any,
        gene_indices: Array,
        kind: InterventionKind = InterventionKind.KNOCKOUT,
        strength: float = 1.0,
    ) -> Any:
        """Return new params with ``gene_indices`` perturbed.

        ``strength`` is the knockdown fraction in ``[0, 1]`` for soft interventions
        (``1.0`` == full knockout); it is ignored for hard knockouts.
        """

    def interventional(
        self,
        params: Any,
        key: Array,
        gene_indices: Array,
        kind: InterventionKind = InterventionKind.KNOCKOUT,
        strength: float = 1.0,
        **kwargs: Any,
    ) -> Array:
        """Intervene on ``gene_indices`` and simulate the perturbed distribution."""
        perturbed = self.intervene(params, gene_indices, kind=kind, strength=strength)
        return self.observational(perturbed, key, **kwargs)
