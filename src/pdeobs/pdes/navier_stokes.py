"""Compact deterministic 2-D incompressible Navier--Stokes generator."""

from __future__ import annotations

from typing import Any

import numpy as np

from .common import (
    PDEOutput,
    Resolution,
    add_channel,
    apply_scalar_boundary,
    apply_velocity_boundary,
    build_output,
    gradient,
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
from .numerics import advance_bounded_velocity, advance_periodic_vorticity

FAMILY = "navier_stokes"
DEFAULT_TIME_STEPS = 9
# Regime labels increase Reynolds-like difficulty, so viscosity decreases.
VISCOSITY_BY_REGIME = {"low": 0.02, "medium": 0.008, "high": 0.0025}
VORTICITY_SCALE_BY_REGIME = {"low": 2.0, "medium": 3.0, "high": 4.0}


def _interior_obstacle(geometry: np.ndarray) -> np.ndarray:
    solid = geometry > 0.5
    solid = solid.copy()
    solid[[0, -1], :] = False
    solid[:, [0, -1]] = False
    return solid


def _velocity_state(
    latent: np.ndarray,
    boundary: str,
    geometry: np.ndarray,
    dx: float,
    dy: float,
    inflow_speed: float,
) -> np.ndarray:
    # A bounded streamfunction construction gives a discretely compatible
    # initial velocity without invoking the periodic Biot--Savart inverse.
    height, width = latent.shape
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    xx, yy = np.meshgrid(x, y)
    streamfunction = latent * np.sin(np.pi * xx) * np.sin(np.pi * yy)
    psi_x, psi_y = gradient(streamfunction, "dirichlet", dx, dy)
    velocity_x, velocity_y = psi_y, -psi_x
    velocity = np.stack((velocity_x, velocity_y), axis=-1)
    apply_velocity_boundary(
        velocity,
        boundary,
        geometry=add_channel(geometry),
        inflow_speed=inflow_speed,
    )
    return velocity


def generate(
    boundary: str = "periodic",
    setting: str = "dipole_vortex_pair",
    regime: str = "medium",
    seed: int = 0,
    resolution: Resolution = 32,
    time_steps: int | None = None,
    *,
    final_time: float = 0.4,
    inflow_speed: float = 0.75,
    dtype: Any = np.float32,
) -> PDEOutput:
    """Generate an incompressible flow with topology-matched solvers.

    Periodic samples expose scalar vorticity, matching spectral flow
    benchmarks.  Wall and B3 obstacle/inflow samples expose two velocity
    channels.  Periodic samples use the FNO-style dealiased vorticity
    pseudospectral integrator.  Bounded samples use a staggered pressure
    projection; no periodic update is hidden under their wall values.
    """

    boundary = normalize_boundary(boundary)
    setting = normalize_setting(setting)
    regime = normalize_regime(regime)
    steps = resolve_time_steps(time_steps, temporal=True)
    height, width = parse_resolution(resolution)
    _, _, dx, dy = grid((height, width))
    if float(final_time) < 0.0:
        raise ValueError("final_time must be non-negative")
    if float(inflow_speed) < 0.0:
        raise ValueError("inflow_speed must be non-negative")

    viscosity = VISCOSITY_BY_REGIME[regime]
    vorticity_scale = VORTICITY_SCALE_BY_REGIME[regime]
    vorticity = vorticity_scale * make_setting_field(
        setting,
        (height, width),
        make_rng(seed, 700),
    )
    if boundary == "periodic":
        vorticity -= float(np.mean(vorticity))
    else:
        apply_scalar_boundary(vorticity, boundary)
    geometry = make_geometry(
        boundary,
        (height, width),
        family=FAMILY,
        rng=make_rng(seed, 701),
    )
    obstacle = _interior_obstacle(geometry)
    if boundary == "periodic":
        initial_encoded = add_channel(vorticity.copy())
    else:
        initial_encoded = _velocity_state(
            vorticity, boundary, geometry, dx, dy, float(inflow_speed)
        )
        initial_encoded[obstacle] = 0.0
    encoded_states = [initial_encoded.copy()]
    total_substeps = 0
    substeps_per_frame: list[int] = []
    max_courant = 0.0
    pressure_iterations_max = 0
    pressure_relative_residual_max = 0.0
    divergence_loss_normalized_solver = 0.0
    xx, yy, _, _ = grid((height, width))
    forcing = 0.1 * (
        np.sin(2.0 * np.pi * (xx + yy)) + np.cos(2.0 * np.pi * (xx + yy))
    )
    if steps > 1:
        frame_dt = float(final_time) / (steps - 1)
        if boundary == "periodic":
            for _ in range(1, steps):
                vorticity, substeps = advance_periodic_vorticity(
                    vorticity, forcing, viscosity, frame_dt, dx, dy
                )
                total_substeps += substeps
                substeps_per_frame.append(substeps)
                encoded_states.append(add_channel(vorticity.copy()))
        else:
            velocity = initial_encoded.copy()
            for _ in range(1, steps):
                speed = np.linalg.norm(velocity, axis=-1)
                max_courant = max(
                    max_courant,
                    float(np.max(speed)) * frame_dt / min(dx, dy),
                )
                velocity, diagnostics = advance_bounded_velocity(
                    velocity,
                    geometry,
                    viscosity,
                    frame_dt,
                    dx,
                    dy,
                    boundary,
                    inflow_speed=float(inflow_speed),
                )
                substeps = int(diagnostics["substeps"])
                total_substeps += substeps
                substeps_per_frame.append(substeps)
                pressure_iterations_max = max(
                    pressure_iterations_max,
                    int(diagnostics["pressure_iterations_max"]),
                )
                pressure_relative_residual_max = max(
                    pressure_relative_residual_max,
                    float(diagnostics["pressure_relative_residual_max"]),
                )
                divergence_loss_normalized_solver = max(
                    divergence_loss_normalized_solver,
                    float(diagnostics["divergence_loss_normalized_solver"]),
                )
                encoded_states.append(velocity.copy())

    trajectory = np.stack(encoded_states, axis=0)
    return build_output(
        family=FAMILY,
        boundary=boundary,
        setting=setting,
        regime=regime,
        seed=seed,
        condition=initial_encoded,
        trajectory=trajectory,
        geometry=add_channel(geometry),
        parameters={
            "viscosity": viscosity,
            "reynolds_proxy": 1.0 / viscosity,
            "vorticity_scale": vorticity_scale,
            "inflow_speed": float(inflow_speed) if boundary == "robin" else 0.0,
            "final_time": float(final_time),
            "time_steps": steps,
            "total_substeps": total_substeps,
            "substeps_per_frame": substeps_per_frame,
            "max_frame_courant": max_courant,
            "substep_cap_hits": 0,
            "clip_count": 0,
            "pressure_iterations_max": pressure_iterations_max,
            "pressure_relative_residual_max": pressure_relative_residual_max,
            "divergence_loss_normalized_solver": divergence_loss_normalized_solver,
            "forcing_id": "fno_sine_cosine_v1" if boundary == "periodic" else "none",
            "forcing_amplitude": 0.1 if boundary == "periodic" else 0.0,
            "integrator_id": (
                "fno_dealiased_vorticity_cn_v2"
                if boundary == "periodic"
                else "mac_projection_fd2_v2"
            ),
            "quality_residual_contract": (
                "pdeobs.quality.navier_stokes.vorticity_forced_fd2_v2"
                if boundary == "periodic"
                else "pdeobs.quality.navier_stokes.velocity_curl_partial_v2"
            ),
            "state_channels": int(trajectory.shape[-1]),
        },
        dtype=dtype,
    )


generate_navier_stokes = generate


__all__ = [
    "DEFAULT_TIME_STEPS",
    "FAMILY",
    "VISCOSITY_BY_REGIME",
    "VORTICITY_SCALE_BY_REGIME",
    "generate",
    "generate_navier_stokes",
]
