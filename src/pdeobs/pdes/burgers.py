"""Compact deterministic two-dimensional scalar Burgers generator."""

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
from .numerics import advance_burgers

FAMILY = "burgers"
DEFAULT_TIME_STEPS = 9
# Regime labels follow increasing Reynolds-like difficulty, hence viscosity falls.
VISCOSITY_BY_REGIME = {"low": 0.04, "medium": 0.015, "high": 0.004}


def generate(
    boundary: str = "dirichlet",
    setting: str = "smooth_grf",
    regime: str = "medium",
    seed: int = 0,
    resolution: Resolution = 32,
    time_steps: int | None = None,
    *,
    final_time: float = 0.35,
    dtype: Any = np.float32,
) -> PDEOutput:
    """Generate ``u_t + u(u_x+u_y) = viscosity*Laplace(u)``.

    Semi-Lagrangian advection and spectral diffusion keep the compact solver
    stable at the small resolutions intended for fixtures and smoke datasets.
    """

    boundary = normalize_boundary(boundary)
    setting = normalize_setting(setting)
    regime = normalize_regime(regime)
    steps = resolve_time_steps(time_steps, temporal=True)
    height, width = parse_resolution(resolution)
    _, _, dx, dy = grid((height, width))
    if float(final_time) < 0.0:
        raise ValueError("final_time must be non-negative")
    viscosity = VISCOSITY_BY_REGIME[regime]
    initial = make_setting_field(setting, (height, width), make_rng(seed, 600))
    apply_scalar_boundary(initial, boundary)

    states = [initial.copy()]
    total_substeps = 0
    substeps_per_frame: list[int] = []
    max_courant = 0.0
    maximum_diffusion_iterations = 0
    if steps > 1:
        frame_dt = float(final_time) / (steps - 1)
        state = initial.copy()
        for _ in range(1, steps):
            state, diagnostics = advance_burgers(
                state, viscosity, frame_dt, dx, dy, boundary
            )
            substeps = int(diagnostics["substeps"])
            max_courant = max(max_courant, float(diagnostics["max_courant"]))
            maximum_diffusion_iterations = max(
                maximum_diffusion_iterations,
                int(diagnostics["max_diffusion_iterations"]),
            )
            substeps_per_frame.append(substeps)
            total_substeps += substeps
            states.append(state.copy())

    trajectory = add_channel(np.stack(states, axis=0))
    geometry = make_geometry(
        boundary,
        (height, width),
        family=FAMILY,
        rng=make_rng(seed, 601),
    )
    return build_output(
        family=FAMILY,
        boundary=boundary,
        setting=setting,
        regime=regime,
        seed=seed,
        condition=add_channel(initial),
        trajectory=trajectory,
        geometry=add_channel(geometry),
        parameters={
            "viscosity": viscosity,
            "reynolds_proxy": 1.0 / viscosity,
            "final_time": float(final_time),
            "time_steps": steps,
            "total_substeps": total_substeps,
            "substeps_per_frame": substeps_per_frame,
            "max_frame_courant": max_courant,
            "substep_cap_hits": 0,
            "clip_count": 0,
            "linear_solver_iterations_max": maximum_diffusion_iterations,
            "integrator_id": (
                "dealiased_pseudospectral_imex_v2"
                if boundary == "periodic"
                else "fd2_rk2_imex_boundary_v2"
            ),
            "quality_residual_contract": "pdeobs.quality.burgers.fd2_saved_frame_v1",
        },
        dtype=dtype,
    )


generate_burgers = generate


__all__ = ["DEFAULT_TIME_STEPS", "FAMILY", "VISCOSITY_BY_REGIME", "generate", "generate_burgers"]
