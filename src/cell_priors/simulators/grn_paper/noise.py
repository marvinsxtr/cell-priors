"""Optional technical noise for the grn-paper simulator (shares SERGIO's model)."""

from __future__ import annotations

from jax import Array

from ..sergio.noise import NoiseProfile, add_technical_noise, resolve_profile


def maybe_add_noise(expr: Array, key: Array, add: bool, profile: str | NoiseProfile = "DS6") -> Array:
    """Apply the shared technical-noise pipeline when ``add`` is True.

    ``profile`` is a DS preset name or an explicit :class:`NoiseProfile`.
    """
    if not add:
        return expr
    return add_technical_noise(key, expr, resolve_profile(profile))
