"""Compact deterministic heat-equation reference generator."""

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
from .numerics import crank_nicolson_diffusion

FAMILY = "heat"
DEFAULT_TIME_STEPS = 9
DIFFUSIVITY_BY_REGIME = {"low": 0.005, "medium": 0.02, "high": 0.08}


def generate(
    boundary: str = "dirichlet",
    setting: str = "smooth_grf",
    regime: str = "medium",
    seed: int = 0,
    resolution: Resolution = 32,
    time_steps: int | None = None,
    *,
    final_time: float = 0.25,
    dtype: Any = np.float32,
) -> PDEOutput:
    """Generate ``u_t = diffusivity * Laplace(u)`` including the initial state."""

    boundary = normalize_boundary(boundary)
    setting = normalize_setting(setting)
    regime = normalize_regime(regime)
    steps = resolve_time_steps(time_steps, temporal=True)
    height, width = parse_resolution(resolution)
    _, _, dx, dy = grid((height, width))
    if float(final_time) < 0.0:
        raise ValueError("final_time must be non-negative")
    diffusivity = DIFFUSIVITY_BY_REGIME[regime]
    initial = make_setting_field(setting, (height, width), make_rng(seed, 400))
    apply_scalar_boundary(initial, boundary)

    states = [initial.copy()]
    maximum_solver_iterations = 0
    maximum_solver_residual = 0.0
    if steps > 1:
        frame_dt = float(final_time) / (steps - 1)
        state = initial.copy()
        for _ in range(1, steps):
            state, solver = crank_nicolson_diffusion(state, diffusivity, frame_dt, dx, dy, boundary)
            maximum_solver_iterations = max(maximum_solver_iterations, solver.iterations)
            maximum_solver_residual = max(maximum_solver_residual, solver.relative_residual)
            states.append(state.copy())
    trajectory = add_channel(np.stack(states, axis=0))
    geometry = make_geometry(
        boundary,
        (height, width),
        family=FAMILY,
        rng=make_rng(seed, 401),
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
            "diffusivity": diffusivity,
            "final_time": float(final_time),
            "time_steps": steps,
            "linear_solver_iterations_max": maximum_solver_iterations,
            "linear_solver_relative_residual_max": maximum_solver_residual,
            "integrator_id": (
                "fourier_exact_diffusion_v2"
                if boundary == "periodic"
                else "fd2_crank_nicolson_boundary_v2"
            ),
            "quality_residual_contract": (
                "pdeobs.quality.heat.post_initial_spectral_plus_replay_v2"
                if boundary == "periodic"
                else "pdeobs.quality.heat.post_initial_fd2_plus_replay_v2"
            ),
        },
        dtype=dtype,
    )


generate_heat = generate


__all__ = ["DEFAULT_TIME_STEPS", "DIFFUSIVITY_BY_REGIME", "FAMILY", "generate", "generate_heat"]
