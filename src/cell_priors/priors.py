"""Named, preconfigured priors (sampler x simulator combinations)."""

from __future__ import annotations

import dataclasses

from .base import ComposedPrior, GRNSampler
from .samplers import GroupedScaleFreeSampler
from .simulators.sergio import SergioConfig, SergioSimulator


class MapPfnPrior(ComposedPrior):
    """A less-opinionated prior: grouped scale-free sampler x cycle-tolerant SERGIO.

    Standard SERGIO is opinionated about its input GRN: it requires a DAG (cycles
    are removed) and master regulators (source genes that carry the basal drive).
    This prior drops both requirements so it supports *any* sampled GRN:

    * the GRN is simulated exactly as drawn -- cycles included (``acyclic=False``);
    * every gene gets its own basal production rate (``require_mrs=False``), so a
      network with no source nodes is still driven.

    It keeps SERGIO's Hill-function production and stochastic dynamics, so it sits
    between the strict SERGIO prior and the grn-paper sigmoid model.
    """

    def __init__(
        self,
        cfg: SergioConfig | None = None,
        sampler: GRNSampler | None = None,
        **kinetics: object,
    ) -> None:
        cfg = dataclasses.replace(cfg or SergioConfig(), require_mrs=False)
        sampler = sampler or GroupedScaleFreeSampler()
        super().__init__(sampler, SergioSimulator(cfg, acyclic=False, **kinetics))
