# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Compact deterministic 2-D incompressible Navier--Stokes generator."""

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
from .numerics import (
    advance_bounded_vorticity,
    advance_masked_vorticity,
    advance_periodic_vorticity,
    apply_masked_vorticity_boundary,
    apply_vorticity_wall_boundary,
    bounded_velocity_from_vorticity,
    solve_masked_streamfunction,
)

FAMILY = "navier_stokes"
DEFAULT_TIME_STEPS = 9
# Regime labels increase Reynolds-like difficulty, so viscosity decreases.
VISCOSITY_BY_REGIME = {"low": 0.02, "medium": 0.008, "high": 0.0025}
VORTICITY_SCALE_BY_REGIME = {"low": 2.0, "medium": 3.0, "high": 4.0}
# Match the reference Fourier Neural Operator data-generation solver.  This is
# the integrator step, independent of the much coarser saved-frame cadence.
PERIODIC_INTERNAL_TIME_STEP = 1.0e-4
SOLVER_BY_BOUNDARY = {
    "periodic": "fno_spectral_vorticity",
    "dirichlet": "dst_no_slip_vorticity",
    "neumann": "dst_free_slip_vorticity",
    "robin": "masked_obstacle_vorticity",
}


def generate(
    boundary: str = "periodic",
    setting: str = "dipole_vortex_pair",
    regime: str = "medium",
    seed: int = 0,
    resolution: Resolution = 32,
    time_steps: int | None = None,
    *,
    final_time: float = 0.4,
    inflow_speed: float = 0.08,
    solver: str | None = None,
    dtype: Any = np.float32,
) -> PDEOutput:
    """Generate an incompressible flow with topology-matched solvers.

    Every route stores scalar vorticity. Periodic samples use the FNO-style
    dealiased pseudospectral integrator. Rectangular wall cases use a
    DST-diagonalized bounded streamfunction, and B3 obstacle cases use a sparse
    streamfunction solve on the true fluid mask with Thom wall vorticity. No
    bounded route hides a periodic update beneath overwritten wall values.
    """

    boundary = normalize_boundary(boundary)
    setting = normalize_setting(setting)
    regime = normalize_regime(regime)
    solver_route = str(solver or SOLVER_BY_BOUNDARY[boundary]).strip().lower()
    if solver_route != SOLVER_BY_BOUNDARY[boundary]:
        raise ValueError(
            f"solver {solver_route!r} is not registered for Navier-Stokes "
            f"boundary {boundary!r}; expected {SOLVER_BY_BOUNDARY[boundary]!r}"
        )
    steps = resolve_time_steps(time_steps, temporal=True)
    height, width = parse_resolution(resolution)
    _, _, dx, dy = grid((height, width))
    if boundary != "periodic":
        dx, dy = 1.0 / (width - 1), 1.0 / (height - 1)
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
    elif boundary in {"dirichlet", "neumann"}:
        apply_scalar_boundary(vorticity, boundary)
    geometry = make_geometry(
        boundary,
        (height, width),
        family=FAMILY,
        rng=make_rng(seed, 701),
    )
    frame_dt = float(final_time) / max(steps - 1, 1)
    if boundary == "periodic":
        initial_encoded = add_channel(vorticity.copy())
        initial_projection: dict[str, float | int] = {}
    elif boundary in {"dirichlet", "neumann"}:
        _, _, initial_streamfunction = bounded_velocity_from_vorticity(vorticity, dx, dy)
        apply_vorticity_wall_boundary(vorticity, initial_streamfunction, boundary, dx, dy)
        initial_encoded = add_channel(vorticity.copy())
        initial_projection = {}
    else:
        _, _, initial_streamfunction = solve_masked_streamfunction(vorticity, geometry, dx, dy)
        apply_masked_vorticity_boundary(vorticity, initial_streamfunction, geometry, dx, dy)
        initial_encoded = add_channel(vorticity.copy())
        initial_projection = {}
    encoded_states = [initial_encoded.copy()]
    total_substeps = 0
    substeps_per_frame: list[int] = []
    max_courant = 0.0
    pressure_iterations_max = int(initial_projection.get("pressure_iterations_max", 0))
    pressure_relative_residual_max = float(
        initial_projection.get("pressure_relative_residual_max", 0.0)
    )
    divergence_loss_normalized_solver = float(
        initial_projection.get("divergence_loss_normalized_solver", 0.0)
    )
    if boundary == "robin":
        xx, yy = np.meshgrid(
            np.linspace(0.0, 1.0, width),
            np.linspace(0.0, 1.0, height),
        )
    else:
        xx, yy, _, _ = grid((height, width))
    forcing = 0.1 * (np.sin(2.0 * np.pi * (xx + yy)) + np.cos(2.0 * np.pi * (xx + yy)))
    if steps > 1:
        if boundary == "periodic":
            for _ in range(1, steps):
                vorticity, substeps = advance_periodic_vorticity(
                    vorticity,
                    forcing,
                    viscosity,
                    frame_dt,
                    dx,
                    dy,
                    internal_dt=PERIODIC_INTERNAL_TIME_STEP,
                )
                total_substeps += substeps
                substeps_per_frame.append(substeps)
                encoded_states.append(add_channel(vorticity.copy()))
        elif boundary in {"dirichlet", "neumann"}:
            for _ in range(1, steps):
                vorticity, diagnostics = advance_bounded_vorticity(
                    vorticity,
                    viscosity,
                    frame_dt,
                    dx,
                    dy,
                    boundary,
                )
                substeps = int(diagnostics["substeps"])
                total_substeps += substeps
                substeps_per_frame.append(substeps)
                max_courant = max(max_courant, float(diagnostics["max_courant"]))
                divergence_loss_normalized_solver = max(
                    divergence_loss_normalized_solver,
                    float(diagnostics["divergence_loss_normalized_solver"]),
                )
                encoded_states.append(add_channel(vorticity.copy()))
        else:
            for _ in range(1, steps):
                vorticity, diagnostics = advance_masked_vorticity(
                    vorticity,
                    geometry,
                    forcing,
                    viscosity,
                    frame_dt,
                    dx,
                    dy,
                )
                substeps = int(diagnostics["substeps"])
                total_substeps += substeps
                substeps_per_frame.append(substeps)
                max_courant = max(max_courant, float(diagnostics["max_courant"]))
                divergence_loss_normalized_solver = max(
                    divergence_loss_normalized_solver,
                    float(diagnostics["divergence_loss_normalized_solver"]),
                )
                encoded_states.append(add_channel(vorticity.copy()))

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
            "forcing_id": (
                "fno_sine_cosine_v1"
                if boundary == "periodic"
                else "bounded_sine_cosine_v1"
                if boundary == "robin"
                else "none"
            ),
            "forcing_amplitude": (
                0.1 if boundary == "periodic" else 0.1 if boundary == "robin" else 0.0
            ),
            "internal_time_step": (PERIODIC_INTERNAL_TIME_STEP if boundary == "periodic" else 0.0),
            "integrator_id": (
                "fno_dealiased_vorticity_cn_v2"
                if boundary == "periodic"
                else "dst_vorticity_streamfunction_ssprk2_v1"
                if boundary in {"dirichlet", "neumann"}
                else "masked_vorticity_streamfunction_ssprk2_v1"
            ),
            "quality_residual_contract": (
                "pdeobs.quality.navier_stokes.post_initial_vorticity_spectral_plus_replay_v3"
                if boundary == "periodic"
                else "pdeobs.quality.navier_stokes.post_initial_vorticity_fd2_plus_replay_v1"
                if boundary in {"dirichlet", "neumann"}
                else "pdeobs.quality.navier_stokes.masked_vorticity_fd2_plus_replay_v1"
            ),
            "state_representation": (
                "vorticity"
                if boundary == "periodic"
                else "bounded_vorticity"
                if boundary in {"dirichlet", "neumann"}
                else "bounded_obstacle_vorticity"
            ),
            "domain_id": (
                "unit_square_node_centered_v1"
                if boundary != "periodic"
                else "unit_square_cell_centered_v1"
            ),
            "boundary_operator_id": (
                f"pdeobs.navier_stokes.{boundary}.vorticity_streamfunction_v1"
                if boundary in {"dirichlet", "neumann"}
                else f"pdeobs.navier_stokes.{boundary}.masked_vorticity_streamfunction_v1"
                if boundary == "robin"
                else "pdeobs.navier_stokes.periodic.spectral_v1"
            ),
            "state_channels": int(trajectory.shape[-1]),
            "solver_route": solver_route,
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
