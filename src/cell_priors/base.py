"""Common interfaces: GRN samplers, simulators, and the priors that combine them.

The generative process for synthetic single-cell data factorizes into two
independent, swappable pieces:

* a **GRN sampler** (:class:`GRNSampler`) draws a *network structure* -- which
  genes regulate which -- as a :class:`GRN`;
* a **simulator** (:class:`Simulator`) turns a :class:`GRN` into expression by
  attaching kinetic parameters and integrating an expression model, and defines
  how interventions act.

A **prior** is any sampler paired with any simulator. :class:`ComposedPrior`
implements the uniform :class:`Prior` API from that pair, so e.g. the grn-paper
grouped scale-free sampler can drive either the SERGIO simulator or the grn-paper
sigmoid-SDE simulator interchangeably.

Everything is a pytree of JAX arrays and all sampling/simulation methods are pure
functions of ``(params, key)``, so a prior composes with a model inside a single
jitted graph with no host round-trip.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import jax
from jax import Array


class InterventionKind(StrEnum):
    """Supported intervention semantics (a simulator decides how each acts)."""

    KNOCKOUT = "knockout"  # hard: remove the gene's regulatory output entirely
    KNOCKDOWN = "knockdown"  # soft (CRISPRi-like): attenuate it by ``strength``


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GRN:
    """A sampled gene-regulatory network *structure* (simulator-agnostic).

    Stored as a sparse list of unique directed edges ``reg -> tar`` with an
    integer ``weight`` (edge multiplicity), plus a per-gene ``group`` label. How
    these edges become kinetic parameters (signs, Hill coefficients, dense
    interaction weights, ...) is the simulator's job.
    """

    reg_idx: Array  # (E,) int: regulator gene index
    tar_idx: Array  # (E,) int: target gene index
    weight: Array  # (E,) float: edge multiplicity / base weight
    group: Array  # (G,) int: group/module label per gene

    @property
    def num_genes(self) -> int:
        return self.group.shape[0]

    @property
    def num_edges(self) -> int:
        return self.reg_idx.shape[0]


class GRNSampler(ABC):
    """Samples a :class:`GRN` structure."""

    @abstractmethod
    def sample(self, key: Array, num_genes: int, **kwargs: Any) -> GRN:
        """Draw a network structure over ``num_genes`` genes."""


class Simulator(ABC):
    """Turns a :class:`GRN` into expression and defines interventions."""

    @abstractmethod
    def build_params(self, grn: GRN, key: Array) -> Any:
        """Attach kinetic parameters to ``grn``, returning a simulator pytree."""

    @abstractmethod
    def simulate(self, params: Any, key: Array, **kwargs: Any) -> Array:
        """Simulate expression, shape ``(num_cells, num_genes)``."""

    @abstractmethod
    def intervene(
        self,
        params: Any,
        gene_indices: Array,
        kind: InterventionKind = InterventionKind.KNOCKOUT,
        strength: float = 1.0,
    ) -> Any:
        """Return new params with ``gene_indices`` perturbed."""


class Prior(ABC):
    """Abstract generative prior over single-cell expression."""

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
        (``1.0`` == full knockout); hard knockouts ignore it.
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


class ComposedPrior(Prior):
    """A prior built from a :class:`GRNSampler` and a :class:`Simulator`.

    This is the explicit "sampler x simulator" composition: ``sample_params``
    draws a structure and hands it to the simulator to parametrize; the remaining
    methods delegate to the simulator.
    """

    def __init__(self, sampler: GRNSampler, simulator: Simulator) -> None:
        self.sampler = sampler
        self.simulator = simulator

    def sample_params(self, key: Array, num_genes: int = 100, sampler_kwargs: dict | None = None, **_: Any) -> Any:
        k_struct, k_kinetics = jax.random.split(key)
        grn = self.sampler.sample(k_struct, num_genes, **(sampler_kwargs or {}))
        return self.simulator.build_params(grn, k_kinetics)

    def observational(self, params: Any, key: Array, **kwargs: Any) -> Array:
        return self.simulator.simulate(params, key, **kwargs)

    def intervene(
        self,
        params: Any,
        gene_indices: Array,
        kind: InterventionKind = InterventionKind.KNOCKOUT,
        strength: float = 1.0,
    ) -> Any:
        return self.simulator.intervene(params, gene_indices, kind=kind, strength=strength)
