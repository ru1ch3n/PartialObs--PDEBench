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
    spectral_diffuse,
)

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
    if steps > 1:
        frame_dt = float(final_time) / (steps - 1)
        state = initial.copy()
        for _ in range(1, steps):
            state = spectral_diffuse(state, diffusivity, frame_dt, dx, dy)
            apply_scalar_boundary(state, boundary)
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
        },
        dtype=dtype,
    )


generate_heat = generate


__all__ = ["DEFAULT_TIME_STEPS", "DIFFUSIVITY_BY_REGIME", "FAMILY", "generate", "generate_heat"]
