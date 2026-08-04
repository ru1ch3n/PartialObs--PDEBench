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
    semi_lagrangian,
    spectral_diffuse,
)

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
    if steps > 1:
        frame_dt = float(final_time) / (steps - 1)
        state = initial.copy()
        for _ in range(1, steps):
            courant = float(np.max(np.abs(state))) * frame_dt / min(dx, dy)
            substeps = max(1, min(24, int(np.ceil(courant / 0.75))))
            dt = frame_dt / substeps
            total_substeps += substeps
            for _ in range(substeps):
                state = semi_lagrangian(state, state, state, dt, dx, dy, boundary)
                state = spectral_diffuse(state, viscosity, dt, dx, dy)
                state = np.clip(state, -2.0, 2.0)
                apply_scalar_boundary(state, boundary)
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
        },
        dtype=dtype,
    )


generate_burgers = generate


__all__ = ["DEFAULT_TIME_STEPS", "FAMILY", "VISCOSITY_BY_REGIME", "generate", "generate_burgers"]
