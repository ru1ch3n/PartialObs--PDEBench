"""Compact deterministic damped-Helmholtz reference generator."""

from __future__ import annotations

from typing import Any

import numpy as np

from .common import (
    PDEOutput,
    Resolution,
    add_channel,
    apply_scalar_boundary,
    build_output,
    grid,
    make_geometry,
    make_rng,
    make_setting_field,
    normalize_boundary,
    normalize_regime,
    normalize_setting,
    parse_resolution,
    resolve_time_steps,
)

FAMILY = "helmholtz"
DEFAULT_TIME_STEPS = 1
WAVENUMBER_BY_REGIME = {"low": 6.0, "medium": 12.0, "high": 18.0}


def _damped_helmholtz(
    source: np.ndarray,
    wavenumber: float,
    dx: float,
    dy: float,
    boundary: str,
    damping_ratio: float,
) -> np.ndarray:
    """Apply a regularized spectral inverse of ``(-Laplace-k^2)``."""

    height, width = source.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(height, d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(width, d=dx)
    kkx, kky = np.meshgrid(kx, ky)
    denominator = kkx**2 + kky**2 - wavenumber**2
    damping = max(damping_ratio * wavenumber**2, 1.0e-6)
    # The real part of 1 / (denominator + i*damping) gives a bounded,
    # resonance-aware response while keeping the benchmark arrays real.
    transfer = denominator / (denominator**2 + damping**2)
    solution = np.fft.ifft2(np.fft.fft2(source) * transfer).real
    apply_scalar_boundary(solution, boundary)
    return solution


def generate(
    boundary: str = "dirichlet",
    setting: str = "smooth_grf",
    regime: str = "medium",
    seed: int = 0,
    resolution: Resolution = 32,
    time_steps: int | None = None,
    *,
    damping_ratio: float = 0.08,
    dtype: Any = np.float32,
) -> PDEOutput:
    """Generate a source and a finite, resonance-aware Helmholtz response."""

    boundary = normalize_boundary(boundary)
    setting = normalize_setting(setting)
    regime = normalize_regime(regime)
    resolve_time_steps(time_steps, temporal=False)
    height, width = parse_resolution(resolution)
    xx, yy, dx, dy = grid((height, width))
    source = make_setting_field(setting, (height, width), make_rng(seed, 300))
    if boundary == "dirichlet":
        source *= np.sin(np.pi * xx) * np.sin(np.pi * yy)
    elif boundary == "robin":
        source *= np.sin(np.pi * yy)
    wavenumber = WAVENUMBER_BY_REGIME[regime]
    solution = _damped_helmholtz(
        source,
        wavenumber,
        dx,
        dy,
        boundary,
        float(damping_ratio),
    )
    geometry = make_geometry(
        boundary,
        (height, width),
        family=FAMILY,
        rng=make_rng(seed, 301),
    )
    return build_output(
        family=FAMILY,
        boundary=boundary,
        setting=setting,
        regime=regime,
        seed=seed,
        condition=add_channel(source),
        trajectory=add_channel(solution)[None, ...],
        geometry=add_channel(geometry),
        parameters={
            "wavenumber": wavenumber,
            "damping_ratio": float(damping_ratio),
            "solver_id": "regularized_real_spectral_transfer_v1",
        },
        dtype=dtype,
    )


generate_helmholtz = generate


__all__ = ["DEFAULT_TIME_STEPS", "FAMILY", "WAVENUMBER_BY_REGIME", "generate", "generate_helmholtz"]
