"""Compact deterministic nonlinear reaction--diffusion generator."""

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

FAMILY = "reaction_diffusion"
DEFAULT_TIME_STEPS = 9
REACTION_RATE_BY_REGIME = {"low": 0.5, "medium": 1.5, "high": 4.0}
DEFAULT_DIFFUSIVITY = 0.008


def generate(
    boundary: str = "dirichlet",
    setting: str = "smooth_grf",
    regime: str = "medium",
    seed: int = 0,
    resolution: Resolution = 32,
    time_steps: int | None = None,
    *,
    final_time: float = 0.8,
    diffusivity: float = DEFAULT_DIFFUSIVITY,
    dtype: Any = np.float32,
) -> PDEOutput:
    """Generate an Allen--Cahn trajectory ``u_t = D Laplace(u)+r(u-u^3)``."""

    boundary = normalize_boundary(boundary)
    setting = normalize_setting(setting)
    regime = normalize_regime(regime)
    steps = resolve_time_steps(time_steps, temporal=True)
    height, width = parse_resolution(resolution)
    _, _, dx, dy = grid((height, width))
    if float(final_time) < 0.0:
        raise ValueError("final_time must be non-negative")
    if float(diffusivity) < 0.0:
        raise ValueError("diffusivity must be non-negative")
    reaction_rate = REACTION_RATE_BY_REGIME[regime]
    initial = 0.8 * make_setting_field(setting, (height, width), make_rng(seed, 500))
    apply_scalar_boundary(initial, boundary)

    states = [initial.copy()]
    maximum_solver_iterations = 0
    maximum_solver_residual = 0.0
    if steps > 1:
        frame_dt = float(final_time) / (steps - 1)
        substeps = max(1, int(np.ceil(reaction_rate * frame_dt / 0.18)))
        dt = frame_dt / substeps
        state = initial.copy()
        for _ in range(1, steps):
            for _ in range(substeps):
                half_decay = np.exp(-reaction_rate * dt)
                denominator = np.sqrt(
                    np.maximum(
                        state**2 + (1.0 - state**2) * half_decay,
                        np.finfo(np.float64).eps,
                    )
                )
                state = state / denominator
                state, solver = crank_nicolson_diffusion(
                    state, float(diffusivity), dt, dx, dy, boundary
                )
                maximum_solver_iterations = max(
                    maximum_solver_iterations, solver.iterations
                )
                maximum_solver_residual = max(
                    maximum_solver_residual, solver.relative_residual
                )
                denominator = np.sqrt(
                    np.maximum(
                        state**2 + (1.0 - state**2) * half_decay,
                        np.finfo(np.float64).eps,
                    )
                )
                state = state / denominator
                apply_scalar_boundary(state, boundary)
            states.append(state.copy())
    else:
        substeps = 0

    trajectory = add_channel(np.stack(states, axis=0))
    geometry = make_geometry(
        boundary,
        (height, width),
        family=FAMILY,
        rng=make_rng(seed, 501),
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
            "reaction_rate": reaction_rate,
            "diffusivity": float(diffusivity),
            "final_time": float(final_time),
            "time_steps": steps,
            "substeps_per_frame": substeps,
            "clip_count": 0,
            "linear_solver_iterations_max": maximum_solver_iterations,
            "linear_solver_relative_residual_max": maximum_solver_residual,
            "integrator_id": (
                "strang_exact_reaction_fourier_diffusion_v2"
                if boundary == "periodic"
                else "strang_exact_reaction_fd2_cn_boundary_v2"
            ),
            "quality_residual_contract": (
                "pdeobs.quality.reaction_diffusion.fd2_saved_frame_v1"
            ),
        },
        dtype=dtype,
    )


generate_reaction_diffusion = generate


__all__ = [
    "DEFAULT_DIFFUSIVITY",
    "DEFAULT_TIME_STEPS",
    "FAMILY",
    "REACTION_RATE_BY_REGIME",
    "generate",
    "generate_reaction_diffusion",
]
