"""Optional technical noise for the grn-paper simulator (shares SERGIO's model)."""

from __future__ import annotations

from jax import Array

from ..sergio.noise import DS_PROFILES, add_technical_noise


def maybe_add_noise(expr: Array, key: Array, add: bool, profile: str = "DS6") -> Array:
    """Apply the shared technical-noise pipeline when ``add`` is True."""
    if not add:
        return expr
    return add_technical_noise(key, expr, DS_PROFILES[profile])
