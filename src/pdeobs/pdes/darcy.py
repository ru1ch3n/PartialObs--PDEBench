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
    solve_poisson_like,
)

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
    coefficient = np.exp(0.5 * np.log(contrast) * latent)
    forcing = np.sin(2.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy)
    forcing += 0.2 * np.sin(4.0 * np.pi * xx + 0.3) * np.sin(2.0 * np.pi * yy)
    forcing -= float(np.mean(forcing))

    iterations = (
        int(solver_steps)
        if solver_steps is not None
        else max(120, min(360, 6 * max(height, width)))
    )
    solution = solve_poisson_like(
        forcing,
        boundary,
        dx,
        dy,
        coefficient=coefficient,
        iterations=iterations,
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
            "solver_steps": iterations,
            "forcing_amplitude": 1.0,
        },
        dtype=dtype,
    )


generate_darcy = generate


__all__ = ["CONTRAST_BY_REGIME", "DEFAULT_TIME_STEPS", "FAMILY", "generate", "generate_darcy"]
