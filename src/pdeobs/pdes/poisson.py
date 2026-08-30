# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Compact deterministic Poisson reference generator."""

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

FAMILY = "poisson"
DEFAULT_TIME_STEPS = 1
AMPLITUDE_BY_REGIME = {"low": 0.5, "medium": 1.0, "high": 2.0}


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
    """Generate a setting-conditioned source and solve ``-Laplace(u) = f``."""

    boundary = normalize_boundary(boundary)
    setting = normalize_setting(setting)
    regime = normalize_regime(regime)
    resolve_time_steps(time_steps, temporal=False)
    height, width = parse_resolution(resolution)
    _, _, dx, dy = grid((height, width))
    amplitude = AMPLITUDE_BY_REGIME[regime]
    source = amplitude * make_setting_field(setting, (height, width), make_rng(seed, 200))
    if boundary == "periodic":
        source -= float(np.mean(source))
    elif boundary == "neumann":
        # The bounded discrete operator solves for the interior nodes and has
        # a constant nullspace.  Store the exact compatible forcing used by
        # that operator so an independent residual audit sees the same BVP.
        source -= float(np.mean(source[1:-1, 1:-1]))
    iterations = (
        int(solver_steps)
        if solver_steps is not None
        else max(120, min(360, 6 * max(height, width)))
    )
    solution, solver = solve_elliptic(
        source,
        boundary,
        dx,
        dy,
        rtol=1.0e-9,
        maxiter=max(iterations, 4000),
    )
    geometry = make_geometry(
        boundary,
        (height, width),
        family=FAMILY,
        rng=make_rng(seed, 201),
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
            "source_amplitude": amplitude,
            "solver_steps": solver.iterations,
            "solver_maxiter": max(iterations, 4000),
            "solver_rtol": 1.0e-9,
            "solver_relative_residual": solver.relative_residual,
            "solver_id": "fd2_flux_krylov_v2",
            "quality_residual_contract": "pdeobs.quality.poisson.fd2_boundary_v1",
        },
        dtype=dtype,
    )


generate_poisson = generate


__all__ = ["AMPLITUDE_BY_REGIME", "DEFAULT_TIME_STEPS", "FAMILY", "generate", "generate_poisson"]
