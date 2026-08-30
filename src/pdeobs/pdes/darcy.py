# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Compact deterministic Darcy-flow reference generator."""

from __future__ import annotations

from typing import Any

import numpy as np

from .common import (
    PDEOutput,
    Resolution,
    add_channel,
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
from .numerics import solve_elliptic

FAMILY = "darcy"
DEFAULT_TIME_STEPS = 1
CONTRAST_BY_REGIME = {"low": 2.0, "medium": 8.0, "high": 32.0}


def generate(
    boundary: str = "dirichlet",
    setting: str = "smooth_grf",
    regime: str = "medium",
    seed: int = 0,
    resolution: Resolution = 32,
    time_steps: int | None = None,
    *,
    solver_steps: int | None = None,
    dtype: Any = np.float32,
) -> PDEOutput:
    """Generate ``-div(a grad(u)) = f`` with a setting-conditioned ``a``.

    The coefficient is log-scaled so its realized max/min ratio is controlled
    by the requested physical regime.  A fixed smooth, zero-mean forcing keeps
    Neumann and periodic samples compatible.
    """

    boundary = normalize_boundary(boundary)
    setting = normalize_setting(setting)
    regime = normalize_regime(regime)
    resolve_time_steps(time_steps, temporal=False)
    height, width = parse_resolution(resolution)
    xx, yy, dx, dy = grid((height, width))

    coefficient_seed = make_rng(seed, 100)
    latent = make_setting_field(setting, (height, width), coefficient_seed)
    contrast = CONTRAST_BY_REGIME[regime]
    latent_min = float(np.min(latent))
    latent_max = float(np.max(latent))
    latent_span = latent_max - latent_min
    if not np.isfinite(latent_span) or latent_span <= 1.0e-12:
        raise RuntimeError(f"setting {setting!r} produced a degenerate Darcy coefficient latent")
    # Setting generators intentionally have different marginal distributions
    # (Gaussian, binary, localized, and discontinuous).  Their standardized
    # values are not bounded by [-1, 1], so exponentiating the raw field makes
    # the realized contrast depend on the setting and can exceed the requested
    # regime by many orders of magnitude.  Range-normalize first and map the
    # endpoints to contrast**(-1/2) and contrast**(+1/2).
    bounded_latent = 2.0 * (latent - latent_min) / latent_span - 1.0
    coefficient = np.exp(0.5 * np.log(contrast) * bounded_latent)
    stored_coefficient = np.asarray(coefficient, dtype=np.dtype(dtype))
    realized_contrast = float(
        np.max(stored_coefficient).astype(np.float64)
        / np.min(stored_coefficient).astype(np.float64)
    )
    forcing = np.sin(2.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy)
    forcing += 0.2 * np.sin(4.0 * np.pi * xx + 0.3) * np.sin(2.0 * np.pi * yy)
    forcing -= float(np.mean(forcing))

    iterations = (
        int(solver_steps)
        if solver_steps is not None
        else max(120, min(360, 6 * max(height, width)))
    )
    solution, solver = solve_elliptic(
        forcing,
        boundary,
        dx,
        dy,
        coefficient=coefficient,
        rtol=1.0e-9,
        maxiter=max(iterations, 6000),
    )
    geometry = make_geometry(
        boundary,
        (height, width),
        family=FAMILY,
        rng=make_rng(seed, 101),
    )
    trajectory = add_channel(solution)[None, ...]
    return build_output(
        family=FAMILY,
        boundary=boundary,
        setting=setting,
        regime=regime,
        seed=seed,
        condition=add_channel(coefficient),
        trajectory=trajectory,
        geometry=add_channel(geometry),
        parameters={
            "coefficient_contrast": contrast,
            "requested_coefficient_contrast": contrast,
            "realized_coefficient_contrast": realized_contrast,
            "solver_steps": solver.iterations,
            "solver_maxiter": max(iterations, 6000),
            "solver_rtol": 1.0e-9,
            "solver_relative_residual": solver.relative_residual,
            "forcing_amplitude": 1.0,
            "forcing_id": "unit_square_sine_mix_v1",
            "solver_id": "finite_volume_flux_krylov_v2",
            "quality_residual_contract": "pdeobs.quality.darcy.fv2_boundary_v1",
        },
        dtype=dtype,
    )


generate_darcy = generate


__all__ = ["CONTRAST_BY_REGIME", "DEFAULT_TIME_STEPS", "FAMILY", "generate", "generate_darcy"]
