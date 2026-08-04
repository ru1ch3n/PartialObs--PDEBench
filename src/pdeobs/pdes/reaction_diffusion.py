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
    spectral_diffuse,
)

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
    if steps > 1:
        frame_dt = float(final_time) / (steps - 1)
        substeps = max(1, int(np.ceil(reaction_rate * frame_dt / 0.18)))
        dt = frame_dt / substeps
        state = initial.copy()
        for _ in range(1, steps):
            for _ in range(substeps):
                state += 0.5 * dt * reaction_rate * (state - state**3)
                state = np.clip(state, -1.25, 1.25)
                state = spectral_diffuse(state, float(diffusivity), dt, dx, dy)
                state += 0.5 * dt * reaction_rate * (state - state**3)
                state = np.clip(state, -1.25, 1.25)
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
