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
    stream_velocity,
)

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
    vorticity: np.ndarray,
    boundary: str,
    geometry: np.ndarray,
    dx: float,
    dy: float,
    inflow_speed: float,
) -> np.ndarray:
    velocity_x, velocity_y = stream_velocity(vorticity, dx, dy)
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
    """Generate a vorticity-form incompressible flow trajectory.

    Periodic samples expose scalar vorticity, matching spectral flow
    benchmarks.  Wall and B3 obstacle/inflow samples expose two velocity
    channels.  Internally all cases use stable semi-Lagrangian vorticity
    transport with a divergence-free spectral velocity reconstruction.
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
    obstacle = (
        _interior_obstacle(geometry) if boundary == "robin" else np.zeros_like(geometry, dtype=bool)
    )
    if np.any(obstacle):
        vorticity[obstacle] = 0.0

    def encode(state: np.ndarray) -> np.ndarray:
        if boundary == "periodic":
            return add_channel(state.copy())
        return _velocity_state(state, boundary, geometry, dx, dy, float(inflow_speed))

    initial_encoded = encode(vorticity)
    encoded_states = [initial_encoded.copy()]
    total_substeps = 0
    if steps > 1:
        frame_dt = float(final_time) / (steps - 1)
        for _ in range(1, steps):
            velocity_x, velocity_y = stream_velocity(vorticity, dx, dy)
            velocity = np.stack((velocity_x, velocity_y), axis=-1)
            apply_velocity_boundary(
                velocity,
                boundary,
                geometry=add_channel(geometry),
                inflow_speed=float(inflow_speed),
            )
            speed = np.sqrt(velocity[..., 0] ** 2 + velocity[..., 1] ** 2)
            courant = float(np.max(speed)) * frame_dt / min(dx, dy)
            substeps = max(1, min(24, int(np.ceil(courant / 0.75))))
            dt = frame_dt / substeps
            total_substeps += substeps
            for _ in range(substeps):
                velocity_x, velocity_y = stream_velocity(vorticity, dx, dy)
                velocity = np.stack((velocity_x, velocity_y), axis=-1)
                apply_velocity_boundary(
                    velocity,
                    boundary,
                    geometry=add_channel(geometry),
                    inflow_speed=float(inflow_speed),
                )
                vorticity = semi_lagrangian(
                    vorticity,
                    velocity[..., 0],
                    velocity[..., 1],
                    dt,
                    dx,
                    dy,
                    boundary,
                )
                vorticity = spectral_diffuse(vorticity, viscosity, dt, dx, dy)
                vorticity = np.clip(vorticity, -20.0, 20.0)
                if boundary == "periodic":
                    vorticity -= float(np.mean(vorticity))
                else:
                    apply_scalar_boundary(vorticity, boundary)
                if np.any(obstacle):
                    vorticity[obstacle] = 0.0
            encoded_states.append(encode(vorticity))

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
