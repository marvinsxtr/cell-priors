"""Debugging and distributional statistics for expression matrices.

Reusable, script-free helpers for the kinds of checks you reach for constantly
when developing a prior: sparsity, NaNs/Infs, dead genes/cells, library-size and
per-gene moments, and simple distributional summaries for comparing priors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ExprStats:
    """Summary statistics / health checks for a ``(cells, genes)`` matrix."""

    n_cells: int
    n_genes: int
    n_nan: int
    n_inf: int
    n_negative: int
    frac_zero: float
    mean: float
    var: float
    max: float
    dead_genes: int  # genes that are zero in every cell
    dead_cells: int  # cells that are zero across every gene
    mean_library_size: float

    def as_dict(self) -> dict:
        return asdict(self)


def summarize(expr: NDArray) -> ExprStats:
    """Compute health/sparsity statistics for an expression matrix."""
    x = np.asarray(expr, dtype=float)
    finite = np.isfinite(x)
    x_safe = np.where(finite, x, 0.0)
    gene_sums = x_safe.sum(axis=0)
    cell_sums = x_safe.sum(axis=1)
    return ExprStats(
        n_cells=x.shape[0],
        n_genes=x.shape[1],
        n_nan=int(np.isnan(x).sum()),
        n_inf=int(np.isinf(x).sum()),
        n_negative=int((x_safe < 0).sum()),
        frac_zero=float((x_safe == 0).mean()),
        mean=float(x_safe.mean()),
        var=float(x_safe.var()),
        max=float(x_safe.max()) if x.size else 0.0,
        dead_genes=int((gene_sums == 0).sum()),
        dead_cells=int((cell_sums == 0).sum()),
        mean_library_size=float(cell_sums.mean()) if x.shape[0] else 0.0,
    )


def assert_healthy(expr: NDArray) -> None:
    """Raise if the matrix has NaNs, Infs or negative values."""
    s = summarize(expr)
    problems = []
    if s.n_nan:
        problems.append(f"{s.n_nan} NaNs")
    if s.n_inf:
        problems.append(f"{s.n_inf} Infs")
    if s.n_negative:
        problems.append(f"{s.n_negative} negative values")
    if problems:
        raise ValueError("Unhealthy expression matrix: " + ", ".join(problems))


def gene_moments(expr: NDArray, log1p: bool = True) -> dict[str, NDArray]:
    """Per-gene mean, variance, dropout rate and Fano factor."""
    x = np.log1p(expr) if log1p else np.asarray(expr, dtype=float)
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    return {
        "mean": mean,
        "var": var,
        "dropout_rate": (np.asarray(expr) == 0).mean(axis=0),
        "fano": var / np.where(mean > 0, mean, 1.0),
    }
