"""Compact deterministic damped-Helmholtz reference generator."""

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

FAMILY = "helmholtz"
DEFAULT_TIME_STEPS = 1
WAVENUMBER_BY_REGIME = {"low": 0.5, "medium": 1.0, "high": 2.0}


def generate(
    boundary: str = "dirichlet",
    setting: str = "smooth_grf",
    regime: str = "medium",
    seed: int = 0,
    resolution: Resolution = 32,
    time_steps: int | None = None,
    *,
    damping_ratio: float = 0.0,
    solver_maxiter: int = 6000,
    dtype: Any = np.float32,
) -> PDEOutput:
    """Solve the nominal Helmholtz BVP with a second-order discrete operator.

    ``damping_ratio`` is retained as a compatibility argument but must be zero;
    the old real part of a damped periodic transfer was not the stated BVP.
    """

    boundary = normalize_boundary(boundary)
    setting = normalize_setting(setting)
    regime = normalize_regime(regime)
    resolve_time_steps(time_steps, temporal=False)
    height, width = parse_resolution(resolution)
    _, _, dx, dy = grid((height, width))
    if float(damping_ratio) != 0.0:
        raise ValueError("the validated Helmholtz path solves the nominal real BVP; damping=0")
    source = make_setting_field(setting, (height, width), make_rng(seed, 300))
    wavenumber = WAVENUMBER_BY_REGIME[regime]
    solution, solver = solve_elliptic(
        source,
        boundary,
        dx,
        dy,
        reaction=-(wavenumber**2),
        rtol=1.0e-9,
        maxiter=int(solver_maxiter),
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
            "damping_ratio": 0.0,
            "solver_steps": solver.iterations,
            "solver_maxiter": int(solver_maxiter),
            "solver_rtol": 1.0e-9,
            "solver_relative_residual": solver.relative_residual,
            "solver_id": "fd2_helmholtz_krylov_v2",
            "quality_residual_contract": "pdeobs.quality.helmholtz.fd2_boundary_v1",
        },
        dtype=dtype,
    )


generate_helmholtz = generate


__all__ = ["DEFAULT_TIME_STEPS", "FAMILY", "WAVENUMBER_BY_REGIME", "generate", "generate_helmholtz"]
