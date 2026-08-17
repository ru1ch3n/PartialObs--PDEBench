"""Scientific and structural quality control for generated PDE datasets.

The quality layer is intentionally separate from model evaluation.  It measures
whether generated arrays satisfy the family equation and its stored constraints;
it does not compare a learned prediction with a target.  Every built-in family
has a discrete residual that can be evaluated from the canonical arrays and
metadata.  The bundled compact solvers remain development references: a finite
residual is a measurement, not evidence that a paper-quality threshold was met.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .schema import Sample, json_safe

QUALITY_SCHEMA_VERSION = "1.1"
BUILTIN_PDE_FAMILIES = (
    "darcy",
    "poisson",
    "helmholtz",
    "heat",
    "reaction_diffusion",
    "burgers",
    "navier_stokes",
)
_EQUATION_PARAMETER_NAMES: dict[str, tuple[str, ...]] = {
    "darcy": (
        "forcing_id",
        "forcing_amplitude",
        "requested_coefficient_contrast",
    ),
    "poisson": ("source_amplitude",),
    "helmholtz": ("wavenumber",),
    "heat": ("final_time", "diffusivity"),
    "reaction_diffusion": ("final_time", "diffusivity", "reaction_rate"),
    "burgers": ("final_time", "viscosity"),
    "navier_stokes": (
        "final_time",
        "viscosity",
        "inflow_speed",
        "vorticity_scale",
        "forcing_id",
        "forcing_amplitude",
    ),
}
_SOLVER_PARAMETER_NAMES: dict[str, tuple[str, ...]] = {
    "darcy": ("solver_steps",),
    "poisson": ("solver_steps",),
    "helmholtz": ("damping_ratio",),
    "burgers": ("advection_scheme",),
}
VALIDATED_SOLVER_FIDELITIES = frozenset(
    {"validated", "validated_reference", "trusted", "trusted_reference"}
)

_EPS = 1.0e-12
_SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")
_DEFAULT_THRESHOLDS: dict[str, float | None] = {
    "finite_fraction_min": 1.0,
    "geometry_binary_max_error_max": 1.0e-6,
    "initial_condition_loss_normalized_max": 1.0e-6,
    "boundary_condition_loss_normalized_max": 1.0e-4,
    "initial_transition_replay_loss_normalized_max": 5.0e-6,
    # PDE and divergence thresholds must be frozen by the release protocol.
    "pde_loss_normalized_max": None,
    "divergence_loss_normalized_max": None,
}


class QualityError(RuntimeError):
    """Raised when quality data cannot be computed or validated."""


class QualityGateError(QualityError):
    """Raised when generation is configured to reject a failed sample."""


def normalize_quality_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the stable quality configuration used by generation and audits."""

    raw = dict(config or {})
    if raw.get("enabled", True) is False:
        raise ValueError("dataset quality reporting cannot be disabled for generated data")
    profile = str(raw.get("profile", "report")).strip().lower()
    if profile not in {"report", "strict", "publication"}:
        raise ValueError("quality profile must be report, strict, or publication")
    supplied = raw.get("thresholds", {})
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, Mapping):
        raise TypeError("quality thresholds must be a mapping")
    thresholds = dict(_DEFAULT_THRESHOLDS)
    for key, value in supplied.items():
        if key not in thresholds:
            raise ValueError(f"unknown quality threshold {key!r}")
        if value is not None:
            value = float(value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"quality threshold {key!r} must be finite and non-negative")
        thresholds[str(key)] = value
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "profile": profile,
        "require_pde_loss": bool(raw.get("require_pde_loss", True)),
        "thresholds": thresholds,
        "calibration_evidence": json_safe(raw.get("calibration_evidence")),
    }


def _solver_evidence_valid(value: Any, metadata: Mapping[str, Any]) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("schema_version") == "pdeobs.numerical-validation/v1"
        and _SHA256_PATTERN.fullmatch(str(value.get("report_sha256", "")))
        and _SHA256_PATTERN.fullmatch(str(value.get("solver_artifact_sha256", "")))
        and value.get("solver_implementation") == metadata.get("solver_implementation")
        and str(value.get("solver_version")) == str(metadata.get("solver_version"))
    )


def _calibration_evidence_threshold(value: Any, calibration_key: str) -> float | None:
    if not isinstance(value, Mapping):
        return None
    threshold_table = value.get("pde_loss_normalized_max_by_key")
    if not isinstance(threshold_table, Mapping):
        return None
    canonical_table = {
        str(key): threshold_table[key]
        for key in sorted(threshold_table, key=lambda item: str(item))
    }
    try:
        table_digest = hashlib.sha256(
            json.dumps(
                canonical_table,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        return None
    try:
        calibrated_threshold = float(threshold_table[calibration_key])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        value.get("schema_version") != "pdeobs.quality-thresholds/v1"
        or str(value.get("table_sha256", "")).lower() != table_digest
        or not np.isfinite(calibrated_threshold)
        or calibrated_threshold < 0.0
    ):
        return None
    return calibrated_threshold


def _required_finite_parameter(
    parameters: Mapping[str, Any],
    name: str,
    *,
    minimum: float | None = None,
    strictly_greater: bool = False,
) -> float:
    if name not in parameters:
        raise QualityError(f"required equation parameter {name!r} is missing")
    try:
        value = float(parameters[name])
    except (TypeError, ValueError) as exc:
        raise QualityError(f"equation parameter {name!r} must be numeric") from exc
    if not np.isfinite(value):
        raise QualityError(f"equation parameter {name!r} must be finite")
    if minimum is not None:
        valid = value > minimum if strictly_greater else value >= minimum
        if not valid:
            comparison = ">" if strictly_greater else ">="
            raise QualityError(f"equation parameter {name!r} must be {comparison} {minimum}")
    return value


def _stable_parameter_context(value: Any) -> Any:
    """Return deterministic JSON data without permitting NaN/Inf tokens."""

    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else f"nonfinite:{number}"
    if isinstance(value, Mapping):
        return {
            str(key): _stable_parameter_context(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_parameter_context(item) for item in value]
    return json_safe(value)


def calibration_key_for_context(context: Mapping[str, Any]) -> str:
    """Hash one canonical quality-calibration stratum."""

    return hashlib.sha256(
        json.dumps(
            json_safe(context),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_boundary(value: Any) -> str:
    token = str(value or "periodic").strip().lower().replace("-", "_")
    aliases = {
        "robin_obstacle": "robin",
        "mixed": "robin",
        "mixed_robin": "robin",
        "no_slip": "dirichlet",
        "free_slip": "neumann",
    }
    return aliases.get(token, token)


def _neighbors(
    values: np.ndarray, boundary: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if boundary == "periodic":
        return (
            np.roll(values, -1, axis=-1),
            np.roll(values, 1, axis=-1),
            np.roll(values, -1, axis=-2),
            np.roll(values, 1, axis=-2),
        )
    pad_width = [(0, 0)] * values.ndim
    pad_width[-2] = (1, 1)
    pad_width[-1] = (1, 1)
    padded = np.pad(values, pad_width, mode="edge")
    base = [slice(None)] * values.ndim
    east = base.copy()
    west = base.copy()
    north = base.copy()
    south = base.copy()
    east[-2], east[-1] = slice(1, -1), slice(2, None)
    west[-2], west[-1] = slice(1, -1), slice(None, -2)
    north[-2], north[-1] = slice(2, None), slice(1, -1)
    south[-2], south[-1] = slice(None, -2), slice(1, -1)
    return padded[tuple(east)], padded[tuple(west)], padded[tuple(north)], padded[tuple(south)]


def _laplacian(values: np.ndarray, boundary: str, dx: float, dy: float) -> np.ndarray:
    east, west, north, south = _neighbors(values, boundary)
    return (east - 2.0 * values + west) / (dx * dx) + (north - 2.0 * values + south) / (dy * dy)


def _gradient(
    values: np.ndarray, boundary: str, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    east, west, north, south = _neighbors(values, boundary)
    return (east - west) / (2.0 * dx), (north - south) / (2.0 * dy)


def _spectral_operators(
    values: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Fourier x/y derivatives and Laplacian on a periodic grid.

    ``values`` may contain arbitrary leading axes (for example saved time
    frames); the final two axes are always interpreted as ``(y, x)``.
    """

    array = np.asarray(values, dtype=np.float64)
    height, width = array.shape[-2:]
    ky = 2.0 * np.pi * np.fft.fftfreq(height, d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(width, d=dx)
    kkx, kky = np.meshgrid(kx, ky)
    transform = np.fft.fft2(array, axes=(-2, -1))
    grad_x = np.fft.ifft2(1j * kkx * transform, axes=(-2, -1)).real
    grad_y = np.fft.ifft2(1j * kky * transform, axes=(-2, -1)).real
    laplace = np.fft.ifft2(
        -(kkx**2 + kky**2) * transform,
        axes=(-2, -1),
    ).real
    return grad_x, grad_y, laplace


def _dealias_periodic(values: np.ndarray) -> np.ndarray:
    """Apply the same rectangular two-thirds filter as the generators."""

    array = np.asarray(values, dtype=np.float64)
    height, width = array.shape[-2:]
    modes_x = np.fft.fftfreq(width) * width
    modes_y = np.fft.fftfreq(height) * height
    keep = (np.abs(modes_y)[:, None] <= height // 3) & (np.abs(modes_x)[None, :] <= width // 3)
    transform = np.fft.fft2(array, axes=(-2, -1))
    return np.fft.ifft2(transform * keep, axes=(-2, -1)).real


def _rusanov_flux_divergence(values: np.ndarray, boundary: str, dx: float, dy: float) -> np.ndarray:
    """Conservative finite-volume divergence used for nonsmooth Burgers data."""

    state = np.asarray(values, dtype=np.float64)
    if boundary == "periodic":
        center = state
        east = np.roll(state, -1, axis=-1)
        west = np.roll(state, 1, axis=-1)
        north = np.roll(state, -1, axis=-2)
        south = np.roll(state, 1, axis=-2)
    else:
        pad_width = [(0, 0)] * state.ndim
        pad_width[-2] = (1, 1)
        pad_width[-1] = (1, 1)
        padded = np.pad(state, pad_width, mode="edge")
        base = [slice(None)] * state.ndim
        center_slice = base.copy()
        east_slice = base.copy()
        west_slice = base.copy()
        north_slice = base.copy()
        south_slice = base.copy()
        center_slice[-2], center_slice[-1] = slice(1, -1), slice(1, -1)
        east_slice[-2], east_slice[-1] = slice(1, -1), slice(2, None)
        west_slice[-2], west_slice[-1] = slice(1, -1), slice(None, -2)
        north_slice[-2], north_slice[-1] = slice(2, None), slice(1, -1)
        south_slice[-2], south_slice[-1] = slice(None, -2), slice(1, -1)
        center = padded[tuple(center_slice)]
        east = padded[tuple(east_slice)]
        west = padded[tuple(west_slice)]
        north = padded[tuple(north_slice)]
        south = padded[tuple(south_slice)]

    def flux(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        physical = 0.25 * (left * left + right * right)
        speed = np.maximum(np.abs(left), np.abs(right))
        return physical - 0.5 * speed * (right - left)

    return (flux(center, east) - flux(west, center)) / dx + (
        flux(center, north) - flux(south, center)
    ) / dy


def _pde_laplacian(values: np.ndarray, boundary: str, dx: float, dy: float) -> np.ndarray:
    if boundary == "periodic":
        return _spectral_operators(values, dx, dy)[2]
    return _laplacian(values, boundary, dx, dy)


def _pde_gradient(
    values: np.ndarray, boundary: str, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    if boundary == "periodic":
        grad_x, grad_y, _ = _spectral_operators(values, dx, dy)
        return grad_x, grad_y
    return _gradient(values, boundary, dx, dy)


def _periodic_channel_gradient(
    values: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Second-order derivatives for periodic x and bounded y.

    The quality mask removes the outer y rows and the obstacle halo, so centered
    y differences are evaluated only where their stencil is physically valid.
    """

    array = np.asarray(values, dtype=np.float64)
    grad_x = (np.roll(array, -1, axis=-1) - np.roll(array, 1, axis=-1)) / (2.0 * dx)
    grad_y = (np.roll(array, -1, axis=-2) - np.roll(array, 1, axis=-2)) / (2.0 * dy)
    return grad_x, grad_y


def _periodic_channel_laplacian(values: np.ndarray, dx: float, dy: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (np.roll(array, -1, axis=-1) - 2.0 * array + np.roll(array, 1, axis=-1)) / dx**2 + (
        np.roll(array, -1, axis=-2) - 2.0 * array + np.roll(array, 1, axis=-2)
    ) / dy**2


def _variable_diffusion(
    values: np.ndarray,
    coefficient: np.ndarray,
    boundary: str,
    dx: float,
    dy: float,
) -> np.ndarray:
    east, west, north, south = _neighbors(values, boundary)
    ae, aw, an, ass = _neighbors(coefficient, boundary)
    ae = 0.5 * (coefficient + ae)
    aw = 0.5 * (coefficient + aw)
    an = 0.5 * (coefficient + an)
    ass = 0.5 * (coefficient + ass)
    return (ae * (east - values) - aw * (values - west)) / (dx * dx) + (
        an * (north - values) - ass * (values - south)
    ) / (dy * dy)


def _masked_values(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if mask is None:
        return array.reshape(-1)
    spatial_mask = np.asarray(mask, dtype=bool)
    if (
        spatial_mask.ndim == 2
        and array.ndim >= 3
        and tuple(array.shape[-3:-1]) == tuple(spatial_mask.shape)
    ):
        expanded = spatial_mask.reshape((1,) * (array.ndim - 3) + spatial_mask.shape + (1,))
    else:
        expanded = spatial_mask
        while expanded.ndim < array.ndim:
            expanded = expanded[None, ...]
    expanded = np.broadcast_to(expanded, array.shape)
    selected = array[expanded]
    return selected


def _rms(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    selected = _masked_values(values, mask)
    if selected.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(selected, dtype=np.float64))))


def _mse(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    selected = _masked_values(values, mask)
    if selected.size == 0:
        return 0.0
    return float(np.mean(np.square(selected, dtype=np.float64)))


def _relative(numerator: float, denominator: float) -> float:
    if denominator <= _EPS:
        return 0.0 if numerator <= _EPS else float(numerator / _EPS)
    return float(numerator / denominator)


def _active_stencil_mask(geometry: np.ndarray, boundary: str) -> np.ndarray:
    """Select cells whose full finite-difference stencil is physically valid."""

    fluid = np.asarray(geometry[..., 0] <= 0.5, dtype=bool)
    active = fluid.copy()
    if boundary != "periodic":
        active[[0, -1], :] = False
        active[:, [0, -1]] = False
    # A derivative at a fluid cell adjacent to a solid obstacle crosses an
    # undefined interface.  Remove the one-cell halo as well as the solid.
    solid = ~fluid
    if np.any(solid):
        halo = solid.copy()
        for axis in (0, 1):
            halo |= np.roll(solid, 1, axis=axis)
            halo |= np.roll(solid, -1, axis=axis)
        active &= ~halo
    return active


def _geometry_contract_error(
    geometry: np.ndarray,
    *,
    family: str,
    boundary: str,
    parameters: Mapping[str, Any],
) -> tuple[float, str]:
    solid = np.asarray(geometry[..., 0] > 0.5, dtype=bool)
    if boundary == "periodic":
        expected_id = "pdeobs.geometry.empty_periodic_v1"
        error = float(np.mean(solid))
    else:
        expected_id = (
            "pdeobs.geometry.outer_wall_circular_obstacle_v1"
            if family == "navier_stokes" and boundary == "robin"
            else "pdeobs.geometry.outer_wall_v1"
        )
        wall = np.zeros_like(solid)
        wall[[0, -1], :] = True
        wall[:, [0, -1]] = True
        wall_error = float(np.mean(~solid[wall])) if np.any(wall) else 1.0
        interior = solid.copy()
        interior[wall] = False
        if expected_id.endswith("circular_obstacle_v1"):
            obstacle_error = 0.0 if np.any(interior) else 1.0
        else:
            obstacle_error = float(np.mean(interior))
        error = max(wall_error, obstacle_error)
    if parameters.get("geometry_protocol_id") != expected_id:
        error = max(error, 1.0)
    return error, expected_id


def _residual_metrics(
    residual: np.ndarray,
    components: Sequence[np.ndarray],
    mask: np.ndarray | None,
) -> dict[str, float]:
    def normalized(
        selected_residual: np.ndarray,
        selected_components: Sequence[np.ndarray],
    ) -> float:
        numerator = _rms(selected_residual, mask)
        denominator = float(sum(_rms(component, mask) for component in selected_components))
        return _relative(numerator, denominator)

    rms = _rms(residual, mask)
    denominator = float(sum(_rms(component, mask) for component in components))
    metrics = {
        "mse": _mse(residual, mask),
        "rms": rms,
        "denominator_rms": denominator,
        "normalized": _relative(rms, denominator),
    }
    array = np.asarray(residual)
    if array.ndim >= 3 and array.shape[0] > 0:

        def select_step(component: np.ndarray, index: int | slice) -> np.ndarray:
            component_array = np.asarray(component)
            if component_array.ndim == array.ndim and component_array.shape[0] == array.shape[0]:
                return component_array[index]
            return component_array

        per_step = [
            normalized(
                array[index], tuple(select_step(component, index) for component in components)
            )
            for index in range(array.shape[0])
        ]
        metrics["first_step_normalized"] = per_step[0]
        metrics["max_step_normalized"] = max(per_step)
        if array.shape[0] > 1:
            metrics["post_initial_normalized"] = normalized(
                array[1:], tuple(select_step(component, slice(1, None)) for component in components)
            )
    return metrics


def _boundary_loss(
    trajectory: np.ndarray,
    geometry: np.ndarray,
    *,
    family: str,
    boundary: str,
    parameters: Mapping[str, Any],
) -> float:
    if boundary == "periodic":
        return 0.0
    values = np.asarray(trajectory, dtype=np.float64)
    scale = _rms(values)
    residuals: list[np.ndarray] = []
    if (
        family == "navier_stokes"
        and values.shape[-1] == 1
        and parameters.get("integrator_id") == "masked_vorticity_streamfunction_ssprk2_v1"
    ):
        from .pdes.numerics import (
            apply_masked_vorticity_boundary,
            solve_masked_streamfunction,
        )

        height, width = values.shape[1:3]
        dx, dy = 1.0 / (width - 1), 1.0 / (height - 1)
        omega = values[..., 0]
        scale = _rms(omega)
        solid = np.asarray(geometry[..., 0] > 0.5, dtype=bool)
        for frame in omega:
            expected = frame.copy()
            _, _, psi = solve_masked_streamfunction(expected, geometry[..., 0], dx, dy)
            apply_masked_vorticity_boundary(expected, psi, geometry[..., 0], dx, dy)
            residuals.append((frame - expected)[solid])
    elif (
        family == "navier_stokes" and values.shape[-1] == 1 and boundary in {"dirichlet", "neumann"}
    ):
        from .pdes.numerics import bounded_velocity_from_vorticity

        height, width = values.shape[1:3]
        dx, dy = 1.0 / (width - 1), 1.0 / (height - 1)
        omega = values[..., 0]
        scale = _rms(omega)
        if boundary == "neumann":
            residuals.extend((omega[:, [0, -1], :], omega[:, :, [0, -1]]))
        else:
            for frame in omega:
                _, _, psi = bounded_velocity_from_vorticity(frame, dx, dy)
                residuals.extend(
                    (
                        frame[0, 1:-1] + 2.0 * psi[1, 1:-1] / dy**2,
                        frame[-1, 1:-1] + 2.0 * psi[-2, 1:-1] / dy**2,
                        frame[1:-1, 0] + 2.0 * psi[1:-1, 1] / dx**2,
                        frame[1:-1, -1] + 2.0 * psi[1:-1, -2] / dx**2,
                    )
                )
    elif (
        family == "navier_stokes"
        and values.shape[-1] == 3
        and parameters.get("integrator_id") == "d2q9_bgk_bounceback_channel_v1"
    ):
        scale = _rms(values[..., :2])
        residuals.extend((values[:, [0, -1], :, :2],))
        solid = np.asarray(geometry[..., 0] > 0.5, dtype=bool).copy()
        solid[:, [0, -1]] = False
        solid[[0, -1], :] = True
        residuals.append(values[:, solid, :2])
    elif (
        family == "navier_stokes"
        and values.shape[-1] == 3
        and parameters.get("integrator_id") == "periodic_channel_fd2_ssprk2_mac_projection_v1"
    ):
        height, width = values.shape[1:3]
        scale = _rms(values[..., :2])
        u = np.zeros((values.shape[0], height, width + 1), dtype=np.float64)
        v = np.zeros((values.shape[0], height + 1, width), dtype=np.float64)
        u[:, :, :-1] = values[..., 0]
        u[:, :, -1] = u[:, :, 0]
        v[:, :-1, :] = values[..., 1]
        solid = np.asarray(geometry[..., 0] > 0.5, dtype=bool).copy()
        solid[:, [0, -1]] = False
        solid[[0, -1], :] = True
        blocked_u = np.zeros((height, width + 1), dtype=bool)
        blocked_v = np.zeros((height + 1, width), dtype=bool)
        blocked_u[:, 1:-1] = solid[:, :-1] | solid[:, 1:]
        blocked_u[:, 0] = solid[:, -1] | solid[:, 0]
        blocked_u[:, -1] = blocked_u[:, 0]
        blocked_v[1:-1, :] = solid[:-1, :] | solid[1:, :]
        blocked_v[[0, -1], :] = True
        residuals.extend(
            (
                u[:, [0, -1], :],
                v[:, [0, -1], :],
                u[:, blocked_u],
                v[:, blocked_v],
                u[:, :, 0] - u[:, :, -1],
            )
        )
    elif family == "navier_stokes" and values.shape[-1] == 3:
        height, width = values.shape[1:3]
        velocity_scale = _rms(values[..., :2])
        scale = velocity_scale
        u = np.zeros((values.shape[0], height, width + 1), dtype=np.float64)
        v = np.zeros((values.shape[0], height + 1, width), dtype=np.float64)
        u[:, :, :-1] = values[..., 0]
        v[:, :-1, :] = values[..., 1]
        if boundary == "dirichlet":
            residuals.extend(
                (
                    u[:, [0, -1], :],
                    v[:, :, [0, -1]],
                    u[:, :, [0, -1]],
                    v[:, [0, -1], :],
                )
            )
        elif boundary == "neumann":
            residuals.extend(
                (
                    u[:, :, [0, -1]],
                    v[:, [0, -1], :],
                    u[:, 0, :] - u[:, 1, :],
                    u[:, -1, :] - u[:, -2, :],
                    v[:, :, 0] - v[:, :, 1],
                    v[:, :, -1] - v[:, :, -2],
                )
            )
        else:
            y = (np.arange(height, dtype=np.float64) + 0.5) / height
            speed = _required_finite_parameter(parameters, "inflow_speed", minimum=0.0)
            profile = 4.0 * speed * y * (1.0 - y)
            profile[[0, -1]] = 0.0
            u[:, :, -1] = profile[None, :]
            residuals.extend(
                (
                    u[:, [0, -1], :],
                    v[:, :, [0, -1]],
                    v[:, [0, -1], :],
                    u[:, :, 0] - profile[None, :],
                    u[:, :, -1] - profile[None, :],
                )
            )
            solid = np.asarray(geometry[..., 0] > 0.5, dtype=bool).copy()
            solid[:, [0, -1]] = False
            solid[[0, -1], :] = True
            blocked_u = np.zeros((height, width + 1), dtype=bool)
            blocked_v = np.zeros((height + 1, width), dtype=bool)
            blocked_u[:, 1:-1] = solid[:, :-1] | solid[:, 1:]
            blocked_v[1:-1, :] = solid[:-1, :] | solid[1:, :]
            residuals.extend((u[:, blocked_u], v[:, blocked_v]))
    elif family == "navier_stokes" and values.shape[-1] == 2:
        if boundary == "dirichlet":
            residuals.extend((values[:, [0, -1], :, :], values[:, :, [0, -1], :]))
        elif boundary == "neumann":
            residuals.extend(
                (
                    values[:, :, [0, -1], 0],
                    values[:, [0, -1], :, 1],
                    values[:, :, 0, 1] - values[:, :, 1, 1],
                    values[:, :, -1, 1] - values[:, :, -2, 1],
                    values[:, 0, :, 0] - values[:, 1, :, 0],
                    values[:, -1, :, 0] - values[:, -2, :, 0],
                )
            )
        else:
            height = values.shape[1]
            y = (np.arange(height, dtype=np.float64) + 0.5) / height
            speed = _required_finite_parameter(parameters, "inflow_speed", minimum=0.0)
            profile = 4.0 * speed * y * (1.0 - y)
            residuals.extend(
                (
                    values[:, [0, -1], 1:-1, :],
                    values[:, :, 0, 0] - profile[None, :],
                    values[:, :, 0, 1],
                    values[:, :, -1, :] - values[:, :, -2, :],
                )
            )
            solid = np.asarray(geometry[..., 0] > 0.5, dtype=bool)
            solid = solid.copy()
            solid[[0, -1], :] = False
            solid[:, [0, -1]] = False
            if np.any(solid):
                residuals.append(values[:, solid, :])
    else:
        scalar = values[..., 0]
        if boundary == "dirichlet":
            residuals.extend((scalar[:, [0, -1], :], scalar[:, :, [0, -1]]))
        elif boundary == "neumann":
            residuals.extend(
                (
                    scalar[:, 0, :] - scalar[:, 1, :],
                    scalar[:, -1, :] - scalar[:, -2, :],
                    scalar[:, :, 0] - scalar[:, :, 1],
                    scalar[:, :, -1] - scalar[:, :, -2],
                )
            )
        else:
            alpha = _required_finite_parameter(
                parameters, "robin_alpha", minimum=0.0, strictly_greater=True
            )
            beta = _required_finite_parameter(parameters, "robin_beta", minimum=0.0)
            factor = beta / (alpha / scalar.shape[-1] + beta)
            residuals.extend(
                (
                    scalar[:, [0, -1], :],
                    scalar[:, :, 0] - factor * scalar[:, :, 1],
                    scalar[:, :, -1] - factor * scalar[:, :, -2],
                )
            )
    flattened = [np.asarray(item, dtype=np.float64).reshape(-1) for item in residuals]
    combined = np.concatenate(flattened) if flattened else np.zeros(1, dtype=np.float64)
    return _relative(_rms(combined), scale)


def _stream_velocity(vorticity: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = vorticity.shape[-2:]
    ky = 2.0 * np.pi * np.fft.fftfreq(height, d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(width, d=dx)
    kkx, kky = np.meshgrid(kx, ky)
    wave2 = kkx**2 + kky**2
    omega_hat = np.fft.fft2(vorticity - np.mean(vorticity, axis=(-2, -1), keepdims=True))
    psi_hat = np.zeros_like(omega_hat, dtype=np.complex128)
    nonzero = wave2 > 0.0
    psi_hat[..., nonzero] = omega_hat[..., nonzero] / wave2[nonzero]
    velocity_x = np.fft.ifft2(1j * kky * psi_hat).real
    velocity_y = np.fft.ifft2(-1j * kkx * psi_hat).real
    return velocity_x, velocity_y


def _darcy_forcing(height: int, width: int) -> np.ndarray:
    dx, dy = 1.0 / width, 1.0 / height
    x = (np.arange(width, dtype=np.float64) + 0.5) * dx
    y = (np.arange(height, dtype=np.float64) + 0.5) * dy
    xx, yy = np.meshgrid(x, y)
    forcing = np.sin(2.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy)
    forcing += 0.2 * np.sin(4.0 * np.pi * xx + 0.3) * np.sin(2.0 * np.pi * yy)
    return forcing - float(np.mean(forcing))


def _helmholtz_transfer_metrics(
    source: np.ndarray,
    solution: np.ndarray,
    *,
    wavenumber: float,
    damping_ratio: float,
    boundary: str,
    dx: float,
    dy: float,
) -> dict[str, float]:
    """Measure fidelity to the compact generator's regularized real transfer."""

    height, width = source.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(height, d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(width, d=dx)
    kkx, kky = np.meshgrid(kx, ky)
    spectral_denominator = kkx**2 + kky**2 - wavenumber**2
    damping = max(damping_ratio * wavenumber**2, 1.0e-6)
    transfer = spectral_denominator / (spectral_denominator**2 + damping**2)
    expected = np.fft.ifft2(np.fft.fft2(source) * transfer).real
    if boundary != "periodic":
        from .pdes.common import apply_scalar_boundary

        apply_scalar_boundary(expected, boundary)
    error = solution - expected
    return {
        "helmholtz_transfer_loss_rms": _rms(error),
        "helmholtz_transfer_loss_normalized": _relative(_rms(error), _rms(expected)),
        "helmholtz_resonance_margin_normalized": _relative(
            float(np.min(np.abs(spectral_denominator))), wavenumber**2
        ),
    }


def _static_pde_loss(
    family: str,
    condition: np.ndarray,
    trajectory: np.ndarray,
    boundary: str,
    parameters: Mapping[str, Any],
    fluid: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[str, dict[str, float]]:
    solution = trajectory[0, ..., 0]
    if parameters.get("domain_id") not in {
        "unit_square_cell_centered_v1",
        "unit_square_node_centered_v1",
    }:
        raise QualityError("stored domain_id is missing or unsupported")
    if family == "poisson":
        source = condition[..., 0]
        operator = -_laplacian(solution, boundary, dx, dy)
        residual = operator - source
        return "-laplace(u)=f", _residual_metrics(residual, (operator, source), fluid)
    if family == "darcy":
        coefficient = condition[..., 0]
        if parameters.get("forcing_id") != "unit_square_sine_mix_v1":
            raise QualityError("Darcy forcing_id is missing or unsupported")
        if np.any(coefficient <= 0.0):
            raise QualityError("Darcy coefficient must be strictly positive")
        forcing_amplitude = _required_finite_parameter(parameters, "forcing_amplitude")
        source = forcing_amplitude * _darcy_forcing(*solution.shape)
        operator = -_variable_diffusion(solution, coefficient, boundary, dx, dy)
        residual = operator - source
        return "-div(a*grad(u))=f", _residual_metrics(residual, (operator, source), fluid)
    if family == "helmholtz":
        source = condition[..., 0]
        wavenumber = _required_finite_parameter(
            parameters, "wavenumber", minimum=0.0, strictly_greater=True
        )
        operator = -_laplacian(solution, boundary, dx, dy) - wavenumber**2 * solution
        residual = operator - source
        return "(-laplace-k^2)u=f", _residual_metrics(residual, (operator, source), fluid)
    raise QualityError(f"no static PDE residual is registered for {family!r}")


def _temporal_pde_loss(
    family: str,
    trajectory: np.ndarray,
    geometry: np.ndarray,
    boundary: str,
    parameters: Mapping[str, Any],
    fluid: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[str, dict[str, float], dict[str, float]]:
    if parameters.get("domain_id") not in {
        "unit_square_cell_centered_v1",
        "unit_square_node_centered_v1",
    }:
        raise QualityError("stored domain_id is missing or unsupported")
    steps = trajectory.shape[0]
    final_time = _required_finite_parameter(
        parameters, "final_time", minimum=0.0, strictly_greater=True
    )
    if steps < 2:
        raise QualityError("temporal PDE residual requires at least two frames and final_time > 0")
    dt = final_time / (steps - 1)
    physics: dict[str, float] = {}

    if family in {"heat", "reaction_diffusion", "burgers"}:
        state = trajectory[..., 0]
        midpoint = 0.5 * (state[:-1] + state[1:])
        time_term = (state[1:] - state[:-1]) / dt
        laplace = _pde_laplacian(midpoint, boundary, dx, dy)
        if family == "heat":
            diffusivity = _required_finite_parameter(parameters, "diffusivity", minimum=0.0)
            diffusion = diffusivity * laplace
            residual = time_term - diffusion
            operator = "u_t=D*laplace(u)"
            components = (time_term, diffusion)
        elif family == "reaction_diffusion":
            diffusivity = _required_finite_parameter(parameters, "diffusivity", minimum=0.0)
            reaction_rate = _required_finite_parameter(parameters, "reaction_rate", minimum=0.0)
            diffusion = diffusivity * laplace
            reaction = reaction_rate * (midpoint - midpoint**3)
            residual = time_term - diffusion - reaction
            operator = "u_t=D*laplace(u)+r*(u-u^3)"
            components = (time_term, diffusion, reaction)
            physics["state_bound_excess"] = float(np.max(np.maximum(np.abs(state) - 1.25, 0.0)))
        else:
            advection_scheme = str(parameters.get("advection_scheme", "")).lower()
            if advection_scheme == "rusanov":
                advection = _rusanov_flux_divergence(midpoint, boundary, dx, dy)
            else:
                grad_x, grad_y = _pde_gradient(midpoint, boundary, dx, dy)
                advection = midpoint * (grad_x + grad_y)
                if boundary == "periodic":
                    advection = _dealias_periodic(advection)
            viscosity = _required_finite_parameter(parameters, "viscosity", minimum=0.0)
            diffusion = viscosity * laplace
            residual = time_term + advection - diffusion
            operator = (
                "u_t+div(u^2/2)=nu*laplace(u)"
                if advection_scheme == "rusanov"
                else "u_t+u*(u_x+u_y)=nu*laplace(u)"
            )
            components = (time_term, advection, diffusion)
        return operator, _residual_metrics(residual, components, fluid), physics

    if family != "navier_stokes":
        raise QualityError(f"no temporal PDE residual is registered for {family!r}")

    viscosity = _required_finite_parameter(parameters, "viscosity", minimum=0.0)
    if trajectory.shape[-1] == 3:
        height, width = trajectory.shape[1:3]
        collocated_lbm = parameters.get("integrator_id") == "d2q9_bgk_bounceback_channel_v1"
        projected_channel = (
            parameters.get("integrator_id") == "periodic_channel_fd2_ssprk2_mac_projection_v1"
        )
        if collocated_lbm:
            velocity_x = trajectory[..., 0]
            velocity_y = trajectory[..., 1]
            u_faces = v_faces = None
        else:
            u_faces = np.zeros((trajectory.shape[0], height, width + 1), dtype=np.float64)
            v_faces = np.zeros((trajectory.shape[0], height + 1, width), dtype=np.float64)
            u_faces[:, :, :-1] = trajectory[..., 0]
            v_faces[:, :-1, :] = trajectory[..., 1]
            if projected_channel:
                u_faces[:, :, -1] = u_faces[:, :, 0]
            elif boundary == "robin":
                y = (np.arange(height, dtype=np.float64) + 0.5) / height
                speed = _required_finite_parameter(parameters, "inflow_speed", minimum=0.0)
                profile = 4.0 * speed * y * (1.0 - y)
                profile[[0, -1]] = 0.0
                u_faces[:, :, -1] = profile[None, :]
            velocity_x = 0.5 * (u_faces[:, :, :-1] + u_faces[:, :, 1:])
            velocity_y = 0.5 * (v_faces[:, :-1, :] + v_faces[:, 1:, :])
        pressure = trajectory[..., 2]
        u_mid = 0.5 * (velocity_x[:-1] + velocity_x[1:])
        v_mid = 0.5 * (velocity_y[:-1] + velocity_y[1:])
        p_mid = 0.5 * (pressure[:-1] + pressure[1:])
        u_t = (velocity_x[1:] - velocity_x[:-1]) / dt
        v_t = (velocity_y[1:] - velocity_y[:-1]) / dt
        derivative_boundary = "periodic_channel" if projected_channel else boundary
        if derivative_boundary == "periodic_channel":
            du_dx, du_dy = _periodic_channel_gradient(u_mid, dx, dy)
            dv_dx, dv_dy = _periodic_channel_gradient(v_mid, dx, dy)
            pressure_x, pressure_y = _periodic_channel_gradient(p_mid, dx, dy)
            diffusion_u = viscosity * _periodic_channel_laplacian(u_mid, dx, dy)
            diffusion_v = viscosity * _periodic_channel_laplacian(v_mid, dx, dy)
        else:
            du_dx, du_dy = _pde_gradient(u_mid, boundary, dx, dy)
            dv_dx, dv_dy = _pde_gradient(v_mid, boundary, dx, dy)
            pressure_x, pressure_y = _pde_gradient(p_mid, boundary, dx, dy)
            diffusion_u = viscosity * _pde_laplacian(u_mid, boundary, dx, dy)
            diffusion_v = viscosity * _pde_laplacian(v_mid, boundary, dx, dy)
        advection_u = u_mid * du_dx + v_mid * du_dy
        advection_v = u_mid * dv_dx + v_mid * dv_dy
        residual = np.stack(
            (
                u_t + advection_u + pressure_x - diffusion_u,
                v_t + advection_v + pressure_y - diffusion_v,
            ),
            axis=-1,
        )
        components = (
            np.stack((u_t, v_t), axis=-1),
            np.stack((advection_u, advection_v), axis=-1),
            np.stack((pressure_x, pressure_y), axis=-1),
            np.stack((diffusion_u, diffusion_v), axis=-1),
        )
        if collocated_lbm or projected_channel:
            expected_forcing_id = (
                "constant_body_force_v1" if projected_channel else "lbm_constant_body_force_v1"
            )
            if parameters.get("forcing_id") != expected_forcing_id:
                raise QualityError("channel momentum residual requires its stored body-force id")
            body_force = np.zeros_like(residual)
            body_force[..., 0] = _required_finite_parameter(parameters, "forcing_amplitude")
            residual -= body_force
            components = (*components, body_force)
        if collocated_lbm:
            divergence_x, _ = _pde_gradient(velocity_x, boundary, dx, dy)
            _, divergence_y = _pde_gradient(velocity_y, boundary, dx, dy)
        else:
            if u_faces is None or v_faces is None:  # pragma: no cover - invariant
                raise QualityError("MAC faces were not reconstructed")
            divergence_x = (u_faces[:, :, 1:] - u_faces[:, :, :-1]) / dx
            divergence_y = (v_faces[:, 1:, :] - v_faces[:, :-1, :]) / dy
        divergence = divergence_x + divergence_y
        gradient_scale = _rms(divergence_x, fluid) + _rms(divergence_y, fluid)
        vorticity = dv_dx - du_dy
        physics.update(
            {
                "divergence_loss_mse": _mse(divergence, fluid),
                "divergence_loss_normalized": _relative(_rms(divergence, fluid), gradient_scale),
                "kinetic_energy_relative_change": _relative(
                    abs(
                        float(np.mean(velocity_x[-1] ** 2 + velocity_y[-1] ** 2))
                        - float(np.mean(velocity_x[0] ** 2 + velocity_y[0] ** 2))
                    ),
                    float(np.mean(velocity_x[0] ** 2 + velocity_y[0] ** 2)),
                ),
                "enstrophy_relative_change": _relative(
                    abs(float(np.mean(vorticity[-1] ** 2)) - float(np.mean(vorticity[0] ** 2))),
                    float(np.mean(vorticity[0] ** 2)),
                ),
            }
        )
        return (
            "u_t+u*grad(u)=-grad(p)+nu*laplace(u), div(u)=0",
            _residual_metrics(residual, components, fluid),
            physics,
        )
    if trajectory.shape[-1] == 1:
        vorticity = trajectory[..., 0]
        if boundary == "periodic":
            velocity_x, velocity_y = _stream_velocity(vorticity, dx, dy)
        elif parameters.get("integrator_id") == "dst_vorticity_streamfunction_ssprk2_v1":
            from .pdes.numerics import bounded_velocity_from_vorticity

            reconstructed = [
                bounded_velocity_from_vorticity(frame, dx, dy)[:2] for frame in vorticity
            ]
            velocity_x = np.stack([item[0] for item in reconstructed])
            velocity_y = np.stack([item[1] for item in reconstructed])
        elif parameters.get("integrator_id") == "masked_vorticity_streamfunction_ssprk2_v1":
            from .pdes.numerics import solve_masked_streamfunction

            reconstructed = [
                solve_masked_streamfunction(frame, geometry[..., 0], dx, dy)[:2]
                for frame in vorticity
            ]
            velocity_x = np.stack([item[0] for item in reconstructed])
            velocity_y = np.stack([item[1] for item in reconstructed])
        else:
            raise QualityError(
                "bounded scalar Navier-Stokes state lacks a registered velocity reconstruction"
            )
    elif trajectory.shape[-1] == 2:
        velocity_x = trajectory[..., 0]
        velocity_y = trajectory[..., 1]
        dv_dx, _ = _pde_gradient(velocity_y, boundary, dx, dy)
        _, du_dy = _pde_gradient(velocity_x, boundary, dx, dy)
        vorticity = dv_dx - du_dy
    else:
        raise QualityError(
            "Navier-Stokes state must be vorticity1, legacy velocity2, or MAC velocity-pressure3"
        )

    midpoint = 0.5 * (vorticity[:-1] + vorticity[1:])
    omega_t = (vorticity[1:] - vorticity[:-1]) / dt
    omega_x, omega_y = _pde_gradient(midpoint, boundary, dx, dy)
    u_mid = 0.5 * (velocity_x[:-1] + velocity_x[1:])
    v_mid = 0.5 * (velocity_y[:-1] + velocity_y[1:])
    advection = u_mid * omega_x + v_mid * omega_y
    if boundary == "periodic":
        advection = _dealias_periodic(advection)
    diffusion = viscosity * _pde_laplacian(midpoint, boundary, dx, dy)
    forcing_id = parameters.get("forcing_id", "none")
    if forcing_id in {"fno_sine_cosine_v1", "bounded_sine_cosine_v1"}:
        height, width = midpoint.shape[1:3]
        if forcing_id == "bounded_sine_cosine_v1":
            x = np.linspace(0.0, 1.0, width)
            y = np.linspace(0.0, 1.0, height)
        else:
            x = (np.arange(width, dtype=np.float64) + 0.5) / width
            y = (np.arange(height, dtype=np.float64) + 0.5) / height
        xx, yy = np.meshgrid(x, y)
        forcing_amplitude = _required_finite_parameter(parameters, "forcing_amplitude", minimum=0.0)
        forcing = forcing_amplitude * (
            np.sin(2.0 * np.pi * (xx + yy)) + np.cos(2.0 * np.pi * (xx + yy))
        )
        residual = omega_t + advection - diffusion - forcing
        components = (omega_t, advection, diffusion, forcing)
    elif forcing_id in {None, "none"}:
        residual = omega_t + advection - diffusion
        components = (omega_t, advection, diffusion)
    else:
        raise QualityError(f"unsupported Navier-Stokes forcing_id {forcing_id!r}")

    du_dx, _ = _pde_gradient(velocity_x, boundary, dx, dy)
    _, dv_dy = _pde_gradient(velocity_y, boundary, dx, dy)
    divergence = du_dx + dv_dy
    gradient_scale = _rms(du_dx, fluid) + _rms(dv_dy, fluid)
    divergence_rms = _rms(divergence, fluid)
    physics.update(
        {
            "divergence_loss_mse": _mse(divergence, fluid),
            "divergence_loss_normalized": _relative(divergence_rms, gradient_scale),
            "kinetic_energy_relative_change": _relative(
                abs(
                    float(np.mean(velocity_x[-1] ** 2 + velocity_y[-1] ** 2))
                    - float(np.mean(velocity_x[0] ** 2 + velocity_y[0] ** 2))
                ),
                float(np.mean(velocity_x[0] ** 2 + velocity_y[0] ** 2)),
            ),
            "enstrophy_relative_change": _relative(
                abs(float(np.mean(vorticity[-1] ** 2)) - float(np.mean(vorticity[0] ** 2))),
                float(np.mean(vorticity[0] ** 2)),
            ),
        }
    )
    return (
        "omega_t+u*omega_x+v*omega_y=nu*laplace(omega)",
        _residual_metrics(residual, components, fluid),
        physics,
    )


def _initial_transition_replay_loss(
    family: str,
    trajectory: np.ndarray,
    geometry: np.ndarray,
    boundary: str,
    parameters: Mapping[str, Any],
    dx: float,
    dy: float,
) -> float:
    """Replay only the first saved transition with the declared integrator."""

    from .pdes.numerics import (
        advance_bounded_mac_state,
        advance_bounded_velocity,
        advance_burgers,
        advance_lbm_channel,
        advance_masked_vorticity,
        advance_periodic_vorticity,
        advance_projected_channel_velocity,
        advance_reaction_diffusion,
        crank_nicolson_diffusion,
        initialize_lbm_distributions,
        lbm_macroscopic,
    )

    if trajectory.shape[0] < 2:
        raise QualityError("initial-transition replay requires at least two frames")
    final_time = _required_finite_parameter(
        parameters, "final_time", minimum=0.0, strictly_greater=True
    )
    frame_dt = final_time / (trajectory.shape[0] - 1)
    if family == "heat":
        expected, _ = crank_nicolson_diffusion(
            trajectory[0, ..., 0],
            _required_finite_parameter(parameters, "diffusivity", minimum=0.0),
            frame_dt,
            dx,
            dy,
            boundary,
        )
        observed = trajectory[1, ..., 0]
    elif family == "reaction_diffusion":
        expected, _ = advance_reaction_diffusion(
            trajectory[0, ..., 0],
            _required_finite_parameter(parameters, "diffusivity", minimum=0.0),
            _required_finite_parameter(parameters, "reaction_rate", minimum=0.0),
            frame_dt,
            dx,
            dy,
            boundary,
        )
        observed = trajectory[1, ..., 0]
    elif family == "burgers":
        expected, _ = advance_burgers(
            trajectory[0, ..., 0],
            _required_finite_parameter(parameters, "viscosity", minimum=0.0),
            frame_dt,
            dx,
            dy,
            boundary,
            advection_scheme=str(parameters.get("advection_scheme") or "") or None,
        )
        observed = trajectory[1, ..., 0]
    elif family == "navier_stokes" and trajectory.shape[-1] == 1:
        if boundary == "periodic":
            if parameters.get("forcing_id") != "fno_sine_cosine_v1":
                raise QualityError("periodic Navier-Stokes replay requires the stored forcing_id")
            height, width = trajectory.shape[1:3]
            x = (np.arange(width, dtype=np.float64) + 0.5) / width
            y = (np.arange(height, dtype=np.float64) + 0.5) / height
            xx, yy = np.meshgrid(x, y)
            amplitude = _required_finite_parameter(parameters, "forcing_amplitude", minimum=0.0)
            forcing = amplitude * (
                np.sin(2.0 * np.pi * (xx + yy)) + np.cos(2.0 * np.pi * (xx + yy))
            )
            expected, _ = advance_periodic_vorticity(
                trajectory[0, ..., 0],
                forcing,
                _required_finite_parameter(parameters, "viscosity", minimum=0.0),
                frame_dt,
                dx,
                dy,
                internal_dt=_required_finite_parameter(
                    parameters,
                    "internal_time_step",
                    minimum=0.0,
                    strictly_greater=True,
                ),
            )
        elif parameters.get("integrator_id") == "dst_vorticity_streamfunction_ssprk2_v1":
            from .pdes.numerics import advance_bounded_vorticity

            expected, _ = advance_bounded_vorticity(
                trajectory[0, ..., 0],
                _required_finite_parameter(parameters, "viscosity", minimum=0.0),
                frame_dt,
                dx,
                dy,
                boundary,
            )
        elif parameters.get("integrator_id") == "masked_vorticity_streamfunction_ssprk2_v1":
            if parameters.get("forcing_id") != "bounded_sine_cosine_v1":
                raise QualityError("masked Navier-Stokes replay requires bounded_sine_cosine_v1")
            height, width = trajectory.shape[1:3]
            xx, yy = np.meshgrid(
                np.linspace(0.0, 1.0, width),
                np.linspace(0.0, 1.0, height),
            )
            amplitude = _required_finite_parameter(parameters, "forcing_amplitude", minimum=0.0)
            forcing = amplitude * (
                np.sin(2.0 * np.pi * (xx + yy)) + np.cos(2.0 * np.pi * (xx + yy))
            )
            expected, _ = advance_masked_vorticity(
                trajectory[0, ..., 0],
                geometry[..., 0],
                forcing,
                _required_finite_parameter(parameters, "viscosity", minimum=0.0),
                frame_dt,
                dx,
                dy,
            )
        else:
            raise QualityError("bounded vorticity replay lacks a registered integrator")
        observed = trajectory[1, ..., 0]
    elif family == "navier_stokes" and trajectory.shape[-1] == 3:
        integrator_id = parameters.get("integrator_id")
        if integrator_id == "periodic_channel_fd2_ssprk2_mac_projection_v1":
            height, width = trajectory.shape[1:3]
            u_faces = np.zeros((height, width + 1), dtype=np.float64)
            v_faces = np.zeros((height + 1, width), dtype=np.float64)
            u_faces[:, :-1] = trajectory[0, ..., 0]
            u_faces[:, -1] = u_faces[:, 0]
            v_faces[:-1, :] = trajectory[0, ..., 1]
            initial_velocity = np.stack(
                (
                    0.5 * (u_faces[:, :-1] + u_faces[:, 1:]),
                    0.5 * (v_faces[:-1, :] + v_faces[1:, :]),
                ),
                axis=-1,
            )
            expected, _, _ = advance_projected_channel_velocity(
                initial_velocity,
                geometry[..., 0],
                _required_finite_parameter(parameters, "viscosity", minimum=0.0),
                frame_dt,
                dx,
                dy,
                body_force_x=_required_finite_parameter(parameters, "forcing_amplitude"),
            )
        elif integrator_id == "d2q9_bgk_bounceback_channel_v1":
            lbm_dt = _required_finite_parameter(
                parameters, "lbm_internal_time_step", minimum=0.0, strictly_greater=True
            )
            relaxation = _required_finite_parameter(
                parameters, "lbm_relaxation_time", minimum=0.5, strictly_greater=True
            )
            substeps_value = parameters.get("substeps_per_frame", [])
            if not isinstance(substeps_value, Sequence) or not substeps_value:
                raise QualityError("LBM replay requires substeps_per_frame")
            substeps = int(substeps_value[0])
            initial_velocity = trajectory[0, ..., :2]
            initial_pressure = trajectory[0, ..., 2]
            distributions = initialize_lbm_distributions(
                initial_velocity, initial_pressure, lbm_dt, dx
            )
            distributions = advance_lbm_channel(
                distributions,
                geometry[..., 0],
                substeps,
                lbm_dt,
                dx,
                relaxation,
                inflow_speed=_required_finite_parameter(parameters, "inflow_speed", minimum=0.0),
                body_force_x=_required_finite_parameter(parameters, "forcing_amplitude"),
            )
            solid = np.asarray(geometry[..., 0] > 0.5, dtype=bool).copy()
            solid[:, [0, -1]] = False
            solid[[0, -1], :] = True
            u, v, pressure = lbm_macroscopic(distributions, solid, lbm_dt, dx)
            expected = np.stack((u, v, pressure), axis=-1)
        else:
            expected, _ = advance_bounded_mac_state(
                trajectory[0],
                geometry[..., 0],
                _required_finite_parameter(parameters, "viscosity", minimum=0.0),
                frame_dt,
                dx,
                dy,
                boundary,
                inflow_speed=_required_finite_parameter(parameters, "inflow_speed", minimum=0.0),
            )
        observed = trajectory[1]
    elif family == "navier_stokes" and trajectory.shape[-1] == 2:
        expected, _ = advance_bounded_velocity(
            trajectory[0],
            geometry[..., 0],
            _required_finite_parameter(parameters, "viscosity", minimum=0.0),
            frame_dt,
            dx,
            dy,
            boundary,
            inflow_speed=_required_finite_parameter(parameters, "inflow_speed", minimum=0.0),
        )
        observed = trajectory[1]
    else:
        raise QualityError(f"no initial-transition replay is registered for {family!r}")
    error = np.asarray(observed, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    return _relative(_rms(error), _rms(observed) + _rms(expected))


def _check_max(value: float | None, threshold: float | None) -> str:
    if value is None:
        return "not_applicable"
    if threshold is None:
        return "reported"
    return "pass" if value <= threshold else "fail"


def evaluate_sample_quality(
    sample: Sample,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one stored sample and return only JSON-compatible scalars."""

    quality_config = normalize_quality_config(config)
    metadata = dict(sample.metadata)
    family = str(metadata.get("pde", metadata.get("family", ""))).strip().lower()
    is_builtin = family in BUILTIN_PDE_FAMILIES
    boundary = _canonical_boundary(metadata.get("boundary"))
    parameters = metadata.get("parameters", {})
    if not isinstance(parameters, Mapping):
        parameters = {}
    condition = np.asarray(sample.condition, dtype=np.float64)
    trajectory = np.asarray(sample.trajectory, dtype=np.float64)
    geometry = np.asarray(sample.geometry, dtype=np.float64)
    height, width = trajectory.shape[1:3]
    if parameters.get("domain_id") == "unit_square_node_centered_v1":
        dx, dy = 1.0 / (width - 1), 1.0 / (height - 1)
    else:
        dx, dy = 1.0 / width, 1.0 / height
    fluid = _active_stencil_mask(geometry, boundary)

    array_contract_errors: list[str] = []
    if geometry.shape[-1] != 1:
        array_contract_errors.append(f"geometry channels={geometry.shape[-1]} (expected 1)")
    stored_representation = metadata.get("state_representation")
    if is_builtin:
        expected_channels = (
            1
            if family == "navier_stokes"
            and boundary == "robin"
            and parameters.get("integrator_id") == "masked_vorticity_streamfunction_ssprk2_v1"
            else 3
            if family == "navier_stokes" and boundary == "robin"
            else 1
        )
        expected_representation = (
            "bounded_obstacle_vorticity"
            if family == "navier_stokes"
            and boundary == "robin"
            and parameters.get("integrator_id") == "masked_vorticity_streamfunction_ssprk2_v1"
            else "collocated_velocity_pressure"
            if family == "navier_stokes" and boundary == "robin"
            else "bounded_vorticity"
            if family == "navier_stokes" and boundary in {"dirichlet", "neumann"}
            else "vorticity"
            if family == "navier_stokes"
            else "scalar"
        )
        if condition.shape[-1] != expected_channels:
            array_contract_errors.append(
                f"condition channels={condition.shape[-1]} (expected {expected_channels})"
            )
        if trajectory.shape[-1] != expected_channels:
            array_contract_errors.append(
                f"trajectory channels={trajectory.shape[-1]} (expected {expected_channels})"
            )
        if family in {"darcy", "poisson", "helmholtz"} and trajectory.shape[0] != 1:
            array_contract_errors.append(f"static trajectory T={trajectory.shape[0]} (expected 1)")
        if (
            family in {"heat", "reaction_diffusion", "burgers", "navier_stokes"}
            and trajectory.shape[0] < 2
        ):
            array_contract_errors.append("temporal trajectory requires T>=2")
        if stored_representation is not None and stored_representation != expected_representation:
            array_contract_errors.append(
                "state_representation="
                f"{stored_representation!r} (expected {expected_representation!r})"
            )

    arrays = (condition, trajectory, geometry)
    finite_count = sum(int(np.count_nonzero(np.isfinite(array))) for array in arrays)
    total_count = sum(int(array.size) for array in arrays)
    finite_fraction = float(finite_count / max(total_count, 1))
    geometry_error = float(np.max(np.minimum(np.abs(geometry), np.abs(geometry - 1.0))))
    if is_builtin:
        geometry_contract_error: float | None
        geometry_contract_error, geometry_protocol_id = _geometry_contract_error(
            geometry,
            family=family,
            boundary=boundary,
            parameters=parameters,
        )
    else:
        geometry_contract_error = None
        geometry_protocol_id = str(parameters.get("geometry_protocol_id", "unregistered"))
    initial_loss: float | None = None
    if trajectory.shape[0] > 1 and condition.shape == trajectory[0].shape:
        initial_loss = _relative(_rms(trajectory[0] - condition), _rms(condition))
    boundary_reason: str | None = None
    try:
        boundary_loss: float | None = _boundary_loss(
            trajectory,
            geometry,
            family=family,
            boundary=boundary,
            parameters=parameters,
        )
    except (QualityError, FloatingPointError, ValueError) as exc:
        boundary_loss = None
        boundary_reason = str(exc)

    active_cells = int(np.count_nonzero(fluid))
    pde_available = is_builtin and active_cells > 0 and not array_contract_errors
    if active_cells == 0:
        pde_reason: str | None = "no active fluid stencil cells"
    elif array_contract_errors:
        pde_reason = "array contract failed: " + "; ".join(array_contract_errors)
    else:
        pde_reason = None
    pde_operator: str | None = None
    pde_metrics: dict[str, float] = {}
    physics: dict[str, float] = {}
    notes: list[str] = []
    if pde_available:
        try:
            if family in {"darcy", "poisson", "helmholtz"}:
                pde_operator, pde_metrics = _static_pde_loss(
                    family,
                    condition,
                    trajectory,
                    boundary,
                    parameters,
                    fluid,
                    dx,
                    dy,
                )
            else:
                pde_operator, pde_metrics, physics = _temporal_pde_loss(
                    family,
                    trajectory,
                    geometry,
                    boundary,
                    parameters,
                    fluid,
                    dx,
                    dy,
                )
            if not all(np.isfinite(float(value)) for value in pde_metrics.values()):
                raise QualityError("PDE residual produced a non-finite metric")
        except (QualityError, FloatingPointError, ValueError) as exc:
            pde_available = False
            pde_reason = str(exc)
    elif active_cells > 0:
        pde_reason = "no family-specific residual is registered"

    initial_transition_replay: float | None = None
    initial_transition_reason: str | None = None
    if is_builtin and family in {"heat", "reaction_diffusion", "burgers", "navier_stokes"}:
        try:
            initial_transition_replay = _initial_transition_replay_loss(
                family,
                trajectory,
                geometry,
                boundary,
                parameters,
                dx,
                dy,
            )
        except (QualityError, FloatingPointError, ValueError, RuntimeError) as exc:
            initial_transition_reason = str(exc)
        else:
            if not np.isfinite(initial_transition_replay):
                initial_transition_reason = "initial-transition replay produced a non-finite loss"
                initial_transition_replay = None

    if (
        pde_available
        and initial_transition_replay is not None
        and "post_initial_normalized" in pde_metrics
    ):
        pde_metrics["all_steps_normalized"] = pde_metrics["normalized"]
        pde_metrics["normalized"] = pde_metrics["post_initial_normalized"]

    first = trajectory[0]
    last = trajectory[-1]
    first_energy = float(np.mean(np.square(first, dtype=np.float64)))
    last_energy = float(np.mean(np.square(last, dtype=np.float64)))
    physics.update(
        {
            "mass_relative_change": _relative(
                abs(float(np.mean(last)) - float(np.mean(first))), _rms(first)
            ),
            "state_energy_relative_change": _relative(
                abs(last_energy - first_energy), first_energy
            ),
            "trajectory_growth_factor": _relative(
                max(_rms(frame) for frame in trajectory), _rms(first)
            ),
            "trajectory_max_abs": float(np.max(np.abs(trajectory))),
        }
    )
    if family == "darcy":
        try:
            requested = _required_finite_parameter(
                parameters,
                "requested_coefficient_contrast",
                minimum=0.0,
                strictly_greater=True,
            )
            realized = _required_finite_parameter(
                parameters,
                "realized_coefficient_contrast",
                minimum=0.0,
                strictly_greater=True,
            )
        except QualityError as exc:
            notes.append(f"Coefficient contrast diagnostic unavailable: {exc}")
        else:
            physics["coefficient_contrast_relative_error"] = _relative(
                abs(realized - requested), requested
            )
    if (
        family == "helmholtz"
        and parameters.get("solver_id") == "regularized_real_spectral_transfer_v1"
        and pde_available
    ):
        try:
            transfer_wavenumber = _required_finite_parameter(
                parameters, "wavenumber", minimum=0.0, strictly_greater=True
            )
            damping_ratio = _required_finite_parameter(parameters, "damping_ratio", minimum=0.0)
        except QualityError as exc:
            notes.append(f"Compact Helmholtz transfer diagnostic unavailable: {exc}")
        else:
            physics.update(
                _helmholtz_transfer_metrics(
                    condition[..., 0],
                    trajectory[0, ..., 0],
                    wavenumber=transfer_wavenumber,
                    damping_ratio=damping_ratio,
                    boundary=boundary,
                    dx=dx,
                    dy=dy,
                )
            )
    for parameter_name, metric_name in (
        ("clip_count", "solver_clip_count"),
        ("max_frame_courant", "solver_max_frame_courant"),
        ("substep_cap_hits", "solver_substep_cap_hits"),
        ("total_substeps", "solver_total_substeps"),
    ):
        if parameter_name in parameters:
            try:
                number = float(parameters[parameter_name])
            except (TypeError, ValueError):
                notes.append(f"Solver diagnostic {parameter_name!r} is non-numeric")
            else:
                if np.isfinite(number):
                    physics[metric_name] = number
                else:
                    notes.append(f"Solver diagnostic {parameter_name!r} is non-finite")
    for metric_name, metric_value in tuple(physics.items()):
        if not np.isfinite(float(metric_value)):
            del physics[metric_name]
            notes.append(f"Physical diagnostic {metric_name!r} was non-finite and omitted")

    pde_loss_status = "measured"
    pde_loss_interpretation = "discrete_saved_field_equation_residual"
    if not pde_available:
        pde_loss_status = "unsupported"
        pde_loss_interpretation = "unavailable"
    elif family == "helmholtz":
        full_bvp_contract = (
            parameters.get("quality_residual_contract")
            == "pdeobs.quality.helmholtz.fd2_boundary_v1"
        )
        pde_loss_status = (
            "partial" if boundary != "periodic" and not full_bvp_contract else "measured"
        )
        pde_loss_interpretation = "nominal_equation_residual_not_regularized_transfer_defect"
        if parameters.get("solver_id") == "regularized_real_spectral_transfer_v1":
            notes.append(
                "The nominal Helmholtz residual and the compact regularized-transfer "
                "defect are reported separately."
            )
        else:
            notes.append(
                "No compact transfer defect was computed because the stored solver_id "
                "does not identify that transfer."
            )
    elif family in {"heat", "reaction_diffusion", "burgers", "navier_stokes"}:
        pde_loss_interpretation = (
            "post_initial_saved_frame_balance_with_separate_initial_transition_replay"
        )
        if initial_transition_reason:
            notes.append("Initial-transition replay unavailable: " + initial_transition_reason)
    if array_contract_errors:
        notes.append("Array contract failed: " + "; ".join(array_contract_errors))
    if boundary_reason:
        notes.append("Boundary diagnostic unavailable: " + boundary_reason)
    if pde_available and family == "navier_stokes" and trajectory.shape[-1] == 2:
        pde_loss_status = "partial"
        notes.append(
            "Bounded velocity storage permits a curl-reconstructed interior diagnostic; "
            "pressure and the hidden transported vorticity are not stored."
        )
    if boundary != "periodic" and family == "navier_stokes":
        if trajectory.shape[-1] == 3:
            if parameters.get("integrator_id") == "periodic_channel_fd2_ssprk2_mac_projection_v1":
                notes.append(
                    "The B3 obstacle state stores MAC face velocities and pressure from "
                    "a streamwise-periodic D2Q9 predictor followed by an explicit "
                    "pressure projection; momentum, discrete divergence, walls, obstacle "
                    "faces, and exact first-frame replay are all measured."
                )
            elif parameters.get("integrator_id") == "d2q9_bgk_bounceback_channel_v1":
                notes.append(
                    "The B3 obstacle state stores collocated velocity and pressure from a "
                    "streamwise-periodic D2Q9 channel; momentum, divergence, walls, and "
                    "bounce-back obstacle values are measured explicitly."
                )
            else:
                notes.append(
                    "The bounded state stores native MAC face velocities and cell pressure; "
                    "momentum and discrete incompressibility are evaluated without inferring "
                    "a hidden pressure."
                )
        elif trajectory.shape[-1] == 1:
            if parameters.get("integrator_id") == "masked_vorticity_streamfunction_ssprk2_v1":
                notes.append(
                    "The obstacle state stores vorticity; the registered sparse "
                    "masked-streamfunction reconstruction and Thom boundary update "
                    "measure the complete fluid-domain vorticity equation, discrete "
                    "incompressibility, walls, and obstacle constraints."
                )
            elif boundary == "neumann":
                notes.append(
                    "The rectangular free-slip state stores vorticity; the registered "
                    "DST streamfunction reconstruction measures the complete vorticity "
                    "equation, discrete incompressibility, and free-slip boundary."
                )
            else:
                notes.append(
                    "The rectangular no-slip state stores vorticity; the registered "
                    "DST streamfunction reconstruction and wall-vorticity update "
                    "measure the complete vorticity equation, discrete "
                    "incompressibility, and no-slip boundary."
                )
        else:
            notes.append(
                "The legacy bounded velocity-only state permits only a curl-reconstructed "
                "PDE diagnostic; pressure is not stored."
            )

    metrics: dict[str, float | None] = {
        "finite_fraction": finite_fraction,
        "geometry_binary_max_error": geometry_error,
        "geometry_contract_error": geometry_contract_error,
        "fluid_fraction": float(np.mean(geometry[..., 0] <= 0.5)),
        "active_stencil_fraction": float(active_cells / max(height * width, 1)),
        "initial_condition_loss_normalized": initial_loss,
        "boundary_condition_loss_normalized": boundary_loss,
        "initial_transition_replay_loss_normalized": initial_transition_replay,
        "pde_loss_mse": pde_metrics.get("mse") if pde_available else None,
        "pde_loss_rms": pde_metrics.get("rms") if pde_available else None,
        "pde_loss_denominator_rms": (pde_metrics.get("denominator_rms") if pde_available else None),
        "pde_loss_normalized": pde_metrics.get("normalized") if pde_available else None,
        "pde_loss_all_steps_normalized": (
            pde_metrics.get("all_steps_normalized") if pde_available else None
        ),
        "pde_loss_first_step_strong_normalized": (
            pde_metrics.get("first_step_normalized") if pde_available else None
        ),
        **physics,
    }
    thresholds = dict(quality_config["thresholds"])
    checks = {
        "finite": (
            "pass" if finite_fraction >= float(thresholds["finite_fraction_min"]) else "fail"
        ),
        "geometry_binary": _check_max(geometry_error, thresholds["geometry_binary_max_error_max"]),
        "geometry_contract": (
            "pass"
            if geometry_contract_error is not None and geometry_contract_error <= _EPS
            else "fail"
            if geometry_contract_error is not None
            else "unavailable"
        ),
        "array_contract": "pass" if not array_contract_errors else "fail",
        "active_domain": "pass" if active_cells > 0 else "fail",
        "initial_condition": _check_max(
            initial_loss, thresholds["initial_condition_loss_normalized_max"]
        ),
        "boundary_condition": (
            _check_max(boundary_loss, thresholds["boundary_condition_loss_normalized_max"])
            if boundary_loss is not None
            else "fail"
        ),
        "pde_loss": (
            _check_max(metrics["pde_loss_normalized"], thresholds["pde_loss_normalized_max"])
            if pde_available
            else ("fail" if quality_config["require_pde_loss"] else "unavailable")
        ),
    }
    if is_builtin and family in {"heat", "reaction_diffusion", "burgers", "navier_stokes"}:
        checks["initial_transition_contract"] = (
            "pass" if initial_transition_replay is not None else "fail"
        )
        checks["initial_transition_replay"] = _check_max(
            initial_transition_replay,
            thresholds["initial_transition_replay_loss_normalized_max"],
        )
    if is_builtin:
        # Built-in generation promises a registered, finite residual even in
        # report mode.  Missing/invalid physical parameters must quarantine the
        # sample instead of silently producing data without its PDE loss.
        checks["pde_residual_contract"] = "pass" if pde_available else "fail"
    divergence = metrics.get("divergence_loss_normalized")
    if divergence is not None:
        checks["incompressibility"] = _check_max(
            divergence, thresholds["divergence_loss_normalized_max"]
        )

    profile = quality_config["profile"]
    solver_fidelity = str(metadata.get("solver_fidelity", "unreported"))
    operator_id: str | None = None
    if is_builtin and family in {"darcy", "poisson", "helmholtz"}:
        operator_id = f"pdeobs.quality.{family}.fd2_v1"
        if family == "helmholtz" and parameters.get("quality_residual_contract"):
            operator_id = str(parameters["quality_residual_contract"])
    elif is_builtin and family == "navier_stokes" and trajectory.shape[-1] == 2:
        operator_id = str(
            parameters.get("quality_residual_contract")
            or "pdeobs.quality.navier_stokes.curl_fd2_saved_frame_partial_v1"
        )
    elif is_builtin:
        declared_contract = parameters.get("quality_residual_contract")
        if declared_contract:
            operator_id = str(declared_contract)
        else:
            spatial_scheme = "spectral" if boundary == "periodic" else "fd2"
            operator_id = f"pdeobs.quality.{family}.midpoint_{spatial_scheme}_saved_frame_v1"
    if trajectory.shape[0] > 1:
        try:
            saved_dt = _required_finite_parameter(
                parameters, "final_time", minimum=0.0, strictly_greater=True
            ) / (trajectory.shape[0] - 1)
        except QualityError:
            saved_dt = None
    else:
        saved_dt = None
    equation_parameters = {
        name: _stable_parameter_context(parameters.get(name))
        for name in _EQUATION_PARAMETER_NAMES.get(family, ())
    }
    residual_protocol = {
        name: _stable_parameter_context(parameters.get(name))
        for name in (
            "domain_id",
            "boundary_operator_id",
            "geometry_protocol_id",
            "quality_residual_contract",
            "robin_alpha",
            "robin_beta",
        )
    }
    solver_parameters = {
        name: _stable_parameter_context(parameters.get(name))
        for name in _SOLVER_PARAMETER_NAMES.get(family, ())
    }
    calibration_context = {
        "pde": family,
        "boundary": str(metadata.get("boundary", boundary)),
        "setting": metadata.get("setting"),
        "regime": metadata.get("regime"),
        "resolution": [height, width],
        "dtype": str(sample.trajectory.dtype),
        "T": int(trajectory.shape[0]),
        "saved_dt": saved_dt,
        "operator_id": operator_id,
        "solver_id": parameters.get("solver_id"),
        "integrator_id": parameters.get("integrator_id"),
        "solver_implementation": metadata.get("solver_implementation"),
        "solver_version": metadata.get("solver_version"),
        "equation_parameters": equation_parameters,
        "solver_parameters": solver_parameters,
        "residual_protocol": residual_protocol,
    }
    calibration_key = calibration_key_for_context(calibration_context)
    calibrated_threshold = _calibration_evidence_threshold(
        quality_config.get("calibration_evidence"), calibration_key
    )
    if profile == "publication":
        notes.append(
            "Solver evidence hashes are treated as an unverified attestation; no trusted "
            "signature/registry is configured in this package."
        )
        thresholds["pde_loss_normalized_max"] = calibrated_threshold
        checks["pde_loss"] = (
            _check_max(metrics["pde_loss_normalized"], calibrated_threshold)
            if pde_available and calibrated_threshold is not None
            else "fail"
        )
        checks["pde_threshold_frozen"] = "pass" if calibrated_threshold is not None else "fail"
        checks["validated_solver"] = (
            "pass" if solver_fidelity in VALIDATED_SOLVER_FIDELITIES else "fail"
        )
        checks["complete_pde_loss"] = "pass" if pde_loss_status == "measured" else "fail"
        checks["solver_validation_evidence"] = (
            "pass"
            if _solver_evidence_valid(metadata.get("solver_validation_evidence"), metadata)
            else "fail"
        )
        # Hash-shaped values in sample metadata are an attestation, not proof
        # that the referenced bytes were independently verified.  This package
        # has no configured trust root/signature registry, so it must not turn
        # that self-report into a publication-candidate pass.
        checks["independent_evidence_verification"] = "fail"
        checks["calibrated_threshold_evidence"] = (
            "pass" if calibrated_threshold is not None else "fail"
        )
    failed = any(value == "fail" for value in checks.values())
    unthresholded = any(value in {"reported", "unavailable"} for value in checks.values())
    status = "fail" if failed else "warning" if unthresholded else "pass"
    sample_quality_gate_ready = False
    return dict(
        json_safe(
            {
                "schema_version": QUALITY_SCHEMA_VERSION,
                "profile": profile,
                "pde": family,
                "boundary": str(metadata.get("boundary", boundary)),
                "operator": pde_operator,
                "operator_id": operator_id,
                "auxiliary_operator_ids": (
                    ["pdeobs.quality.helmholtz.regularized_real_transfer_v1"]
                    if "helmholtz_transfer_loss_normalized" in metrics
                    else []
                ),
                "calibration_key": calibration_key,
                "calibration_context": calibration_context,
                "calibration_evidence": (
                    {
                        "schema_version": quality_config["calibration_evidence"].get(
                            "schema_version"
                        ),
                        "table_sha256": quality_config["calibration_evidence"].get("table_sha256"),
                        "evidence_id": quality_config["calibration_evidence"].get("evidence_id"),
                    }
                    if isinstance(quality_config.get("calibration_evidence"), Mapping)
                    else None
                ),
                "resolution": [height, width],
                "stored_dtype": str(sample.trajectory.dtype),
                "active_spatial_cells": active_cells,
                "geometry_protocol_id": geometry_protocol_id,
                "pde_loss": {
                    "available": pde_available,
                    "status": pde_loss_status,
                    "interpretation": pde_loss_interpretation,
                    **pde_metrics,
                    "reason": pde_reason,
                },
                "metrics": metrics,
                "checks": checks,
                "thresholds": thresholds,
                "status": status,
                "solver_fidelity": solver_fidelity,
                "solver": {
                    "fidelity": solver_fidelity,
                    "version": metadata.get("solver_version"),
                    "implementation": metadata.get("solver_implementation"),
                    "solver_id": parameters.get("solver_id"),
                    "integrator_id": parameters.get("integrator_id"),
                },
                "sample_quality_gate_ready": sample_quality_gate_ready,
                "sample_quality_attestation_complete": bool(
                    profile == "publication"
                    and checks.get("solver_validation_evidence") == "pass"
                    and checks.get("calibrated_threshold_evidence") == "pass"
                    and all(
                        value != "fail"
                        for name, value in checks.items()
                        if name != "independent_evidence_verification"
                    )
                ),
                # Dataset publication additionally needs canonical factor/plan
                # coverage and independently verified release evidence.
                "publication_ready": False,
                "notes": notes,
            }
        )
    )


def generation_quality_rejected(quality: Mapping[str, Any]) -> bool:
    """Return whether generation must quarantine this quality record."""

    profile = str(quality.get("profile", "report"))
    status = str(quality.get("status", "unknown"))
    checks = quality.get("checks", {})
    hard_contract_checks = (
        "finite",
        "geometry_binary",
        "geometry_contract",
        "array_contract",
        "active_domain",
        "pde_residual_contract",
        "initial_transition_contract",
    )
    if isinstance(checks, Mapping) and any(
        checks.get(name) == "fail" for name in hard_contract_checks
    ):
        return True
    return (profile == "strict" and status == "fail") or (
        profile == "publication" and status != "pass"
    )


def enforce_generation_quality(quality: Mapping[str, Any]) -> None:
    """Reject a generated sample when a strict/publication profile fails."""

    if generation_quality_rejected(quality):
        profile = str(quality.get("profile", "report"))
        failed = sorted(
            str(name) for name, value in dict(quality.get("checks", {})).items() if value == "fail"
        )
        raise QualityGateError(
            f"{profile} dataset-quality gate failed: {', '.join(failed) or 'unknown check'}"
        )


@dataclass(slots=True)
class _RunningMetric:
    count: int = 0
    mean_value: float = 0.0
    m2: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    maximum_sample_id: str | None = None

    def update(self, value: Any, *, sample_id: str | None = None) -> None:
        if value is None or isinstance(value, bool):
            return
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(number):
            return
        self.count += 1
        delta = number - self.mean_value
        self.mean_value += delta / self.count
        self.m2 += delta * (number - self.mean_value)
        self.minimum = min(self.minimum, number)
        if number > self.maximum:
            self.maximum = number
            self.maximum_sample_id = sample_id

    def summary(self) -> dict[str, float | int | str | None] | None:
        if self.count == 0:
            return None
        variance = max(self.m2 / self.count, 0.0)
        result: dict[str, float | int | str | None] = {
            "count": self.count,
            "mean": self.mean_value,
            "std": float(np.sqrt(variance)),
            "min": self.minimum,
            "max": self.maximum,
        }
        if self.maximum_sample_id is not None:
            result["max_sample_id"] = self.maximum_sample_id
        return result


class QualityAccumulator:
    """Streaming aggregation of sample-level quality metadata."""

    def __init__(self) -> None:
        self.input_count = 0
        self.record_count = 0
        self.missing_quality_count = 0
        self.invalid_quality_count = 0
        self.statuses: Counter[str] = Counter()
        self.profiles: Counter[str] = Counter()
        self.by_pde_count: Counter[str] = Counter()
        self.by_pde_statuses: dict[str, Counter[str]] = {}
        self.by_pde_loss_statuses: dict[str, Counter[str]] = {}
        self.by_pde_checks: dict[str, dict[str, Counter[str]]] = {}
        self.operators: dict[str, set[str]] = {}
        self.operator_ids: dict[str, set[str]] = {}
        self.calibration_keys: dict[str, set[str]] = {}
        self.calibration_contexts: dict[str, dict[str, Any]] = {}
        self.calibration_thresholds: dict[str, float | None] = {}
        self.schema_versions: Counter[str] = Counter()
        self.sample_quality_gate_ready_count = 0
        self.solver_fidelities: dict[str, Counter[str]] = {}
        self.metrics: dict[str, dict[str, _RunningMetric]] = {}
        self.stratum_counts: Counter[str] = Counter()
        self.stratum_pde_losses: dict[str, _RunningMetric] = {}

    def update(self, row: Mapping[str, Any]) -> bool:
        self.input_count += 1
        if "quality" in row:
            quality = row.get("quality")
            if quality is None:
                self.missing_quality_count += 1
                return False
        elif "metrics" in row:
            quality = row
        else:
            self.missing_quality_count += 1
            return False
        if (
            not isinstance(quality, Mapping)
            or quality.get("schema_version") != QUALITY_SCHEMA_VERSION
            or not isinstance(quality.get("metrics"), Mapping)
            or not isinstance(quality.get("pde_loss"), Mapping)
            or not str(quality.get("calibration_key", "")).strip()
        ):
            self.invalid_quality_count += 1
            return False
        family = str(quality.get("pde", row.get("pde", "unknown")))
        sample_id = str(row.get("sample_id")) if row.get("sample_id") is not None else None
        calibration_key = str(quality.get("calibration_key"))
        calibration_context = quality.get("calibration_context")
        if not isinstance(calibration_context, Mapping):
            self.invalid_quality_count += 1
            return False
        normalized_context = dict(json_safe(calibration_context))
        try:
            computed_key = calibration_key_for_context(normalized_context)
        except (TypeError, ValueError):
            self.invalid_quality_count += 1
            return False
        if computed_key != calibration_key:
            self.invalid_quality_count += 1
            return False
        existing_context = self.calibration_contexts.get(calibration_key)
        if existing_context is not None and existing_context != normalized_context:
            self.invalid_quality_count += 1
            return False
        self.calibration_contexts[calibration_key] = normalized_context
        threshold_payload = quality.get("thresholds", {})
        threshold = (
            threshold_payload.get("pde_loss_normalized_max")
            if isinstance(threshold_payload, Mapping)
            else None
        )
        try:
            normalized_threshold = float(threshold) if threshold is not None else None
        except (TypeError, ValueError):
            self.invalid_quality_count += 1
            return False
        if normalized_threshold is not None and (
            not np.isfinite(normalized_threshold) or normalized_threshold < 0.0
        ):
            self.invalid_quality_count += 1
            return False
        existing_threshold = self.calibration_thresholds.get(calibration_key)
        if (
            calibration_key in self.calibration_thresholds
            and existing_threshold != normalized_threshold
        ):
            self.invalid_quality_count += 1
            return False
        self.calibration_thresholds[calibration_key] = normalized_threshold
        self.record_count += 1
        self.schema_versions[str(quality.get("schema_version"))] += 1
        quality_status = str(quality.get("status", "unknown"))
        self.statuses[quality_status] += 1
        self.profiles[str(quality.get("profile", "unknown"))] += 1
        if quality.get("sample_quality_gate_ready") is True:
            self.sample_quality_gate_ready_count += 1
        self.by_pde_count[family] += 1
        self.by_pde_statuses.setdefault(family, Counter())[quality_status] += 1
        pde_loss = quality.get("pde_loss", {})
        if isinstance(pde_loss, Mapping):
            loss_status = str(pde_loss.get("status", "unknown"))
            self.by_pde_loss_statuses.setdefault(family, Counter())[loss_status] += 1
        checks = quality.get("checks", {})
        if isinstance(checks, Mapping):
            family_checks = self.by_pde_checks.setdefault(family, {})
            for name, value in checks.items():
                family_checks.setdefault(str(name), Counter())[str(value)] += 1
        operator = quality.get("operator")
        if operator:
            self.operators.setdefault(family, set()).add(str(operator))
        operator_id = quality.get("operator_id")
        if operator_id:
            self.operator_ids.setdefault(family, set()).add(str(operator_id))
        self.calibration_keys.setdefault(family, set()).add(calibration_key)
        self.stratum_counts[calibration_key] += 1
        fidelity = str(quality.get("solver_fidelity", row.get("solver_fidelity", "unreported")))
        self.solver_fidelities.setdefault(family, Counter())[fidelity] += 1
        family_metrics = self.metrics.setdefault(family, {})
        metrics = quality.get("metrics", {})
        if isinstance(metrics, Mapping):
            for name, value in metrics.items():
                family_metrics.setdefault(str(name), _RunningMetric()).update(
                    value, sample_id=sample_id
                )
            self.stratum_pde_losses.setdefault(calibration_key, _RunningMetric()).update(
                metrics.get("pde_loss_normalized"), sample_id=sample_id
            )
        return True

    def summary(self, *, expected_families: Sequence[str] = BUILTIN_PDE_FAMILIES) -> dict[str, Any]:
        expected = tuple(str(name) for name in expected_families)
        by_pde: dict[str, Any] = {}
        for family in sorted(set(expected) | set(self.by_pde_count)):
            metric_rows = {
                name: summary
                for name, state in sorted(self.metrics.get(family, {}).items())
                if (summary := state.summary()) is not None
            }
            by_pde[family] = {
                "status": "present" if self.by_pde_count[family] else "missing",
                "sample_count": int(self.by_pde_count[family]),
                "quality_status_counts": dict(
                    sorted(self.by_pde_statuses.get(family, Counter()).items())
                ),
                "pde_loss_status_counts": dict(
                    sorted(self.by_pde_loss_statuses.get(family, Counter()).items())
                ),
                "check_counts": {
                    check: dict(sorted(counts.items()))
                    for check, counts in sorted(self.by_pde_checks.get(family, {}).items())
                },
                "operators": sorted(self.operators.get(family, set())),
                "operator_ids": sorted(self.operator_ids.get(family, set())),
                "calibration_keys": sorted(self.calibration_keys.get(family, set())),
                "solver_fidelities": dict(
                    sorted(self.solver_fidelities.get(family, Counter()).items())
                ),
                "metrics": metric_rows,
            }
        present = [family for family in expected if self.by_pde_count[family] > 0]
        missing = [family for family in expected if self.by_pde_count[family] == 0]
        pde_losses = {
            family: {
                "status": by_pde[family]["status"],
                "sample_count": by_pde[family]["sample_count"],
                "operators": by_pde[family]["operators"],
                "operator_ids": by_pde[family]["operator_ids"],
                "calibration_keys": by_pde[family]["calibration_keys"],
                "pde_loss_mse": by_pde[family]["metrics"].get("pde_loss_mse"),
                "pde_loss_normalized": by_pde[family]["metrics"].get("pde_loss_normalized"),
            }
            for family in expected
        }
        return {
            "schema_version": QUALITY_SCHEMA_VERSION,
            "input_count": self.input_count,
            "record_count": self.record_count,
            "missing_quality_count": self.missing_quality_count,
            "invalid_quality_count": self.invalid_quality_count,
            "status_counts": dict(sorted(self.statuses.items())),
            "profile_counts": dict(sorted(self.profiles.items())),
            "schema_versions": dict(sorted(self.schema_versions.items())),
            "sample_quality_gate_ready_count": self.sample_quality_gate_ready_count,
            "expected_pdes": list(expected),
            "present_pdes": present,
            "missing_pdes": missing,
            "complete_pde_coverage": not missing,
            "pde_losses": pde_losses,
            "by_pde": by_pde,
            "by_calibration_key": {
                key: {
                    "sample_count": int(count),
                    "calibration_context": self.calibration_contexts[key],
                    "pde_loss_normalized_max": self.calibration_thresholds.get(key),
                    "pde_loss_normalized": self.stratum_pde_losses[key].summary(),
                }
                for key, count in sorted(self.stratum_counts.items())
            },
        }


def summarize_quality_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_families: Sequence[str] = BUILTIN_PDE_FAMILIES,
) -> dict[str, Any]:
    accumulator = QualityAccumulator()
    for row in rows:
        accumulator.update(row)
    return accumulator.summary(expected_families=expected_families)


def assess_quality_gate(
    summary: Mapping[str, Any],
    *,
    strict: bool = False,
    max_pde_loss: float | None = None,
    require_all_pdes: bool = False,
    require_validated_solvers: bool = False,
    expected_record_count: int | None = None,
) -> dict[str, Any]:
    """Assess a dataset summary without inventing an unfrozen PDE threshold."""

    if max_pde_loss is not None:
        max_pde_loss = float(max_pde_loss)
        if not np.isfinite(max_pde_loss) or max_pde_loss < 0.0:
            raise ValueError("max_pde_loss must be finite and non-negative")
    reasons: list[str] = []
    record_count = int(summary.get("record_count", 0))
    missing_quality_count = int(summary.get("missing_quality_count", 0))
    invalid_quality_count = int(summary.get("invalid_quality_count", 0))
    if strict and record_count == 0:
        reasons.append("no sample-level quality records were found")
    if strict and missing_quality_count:
        reasons.append(f"{missing_quality_count} samples have no quality record")
    if strict and invalid_quality_count:
        reasons.append(f"{invalid_quality_count} samples have an invalid quality record")
    if expected_record_count is not None and record_count != int(expected_record_count):
        reasons.append(f"quality coverage is {record_count}/{int(expected_record_count)} samples")
    status_counts = summary.get("status_counts", {})
    if strict and int(dict(status_counts).get("fail", 0)) > 0:
        reasons.append(f"{dict(status_counts).get('fail')} samples failed stored checks")
    if require_all_pdes and summary.get("missing_pdes"):
        reasons.append("missing PDE families: " + ", ".join(summary["missing_pdes"]))
    if require_validated_solvers:
        profiles = set(dict(summary.get("profile_counts", {})))
        if profiles != {"publication"}:
            reasons.append(
                "publication-candidate audit requires stored publication-profile records"
            )
        nonpassing = record_count - int(summary.get("sample_quality_gate_ready_count", 0))
        if nonpassing:
            reasons.append(f"{nonpassing} samples did not pass every publication-candidate check")
        nonpass_statuses = {
            str(name): int(count)
            for name, count in dict(status_counts).items()
            if str(name) != "pass" and int(count) > 0
        }
        if nonpass_statuses:
            reasons.append(
                "publication-candidate records include non-pass statuses: "
                + ", ".join(f"{name}={count}" for name, count in sorted(nonpass_statuses.items()))
            )
    by_pde = summary.get("by_pde", {})
    if isinstance(by_pde, Mapping):
        for family, row in by_pde.items():
            if not isinstance(row, Mapping) or int(row.get("sample_count", 0)) == 0:
                continue
            metrics = row.get("metrics", {})
            sample_count = int(row.get("sample_count", 0))
            if max_pde_loss is not None and isinstance(metrics, Mapping):
                loss = metrics.get("pde_loss_normalized")
                observed = loss.get("max") if isinstance(loss, Mapping) else None
                measured_count = int(loss.get("count", 0)) if isinstance(loss, Mapping) else 0
                if observed is None:
                    reasons.append(f"{family}: normalized PDE loss is unavailable")
                elif measured_count != sample_count:
                    reasons.append(
                        f"{family}: normalized PDE loss coverage is "
                        f"{measured_count}/{sample_count} samples"
                    )
                elif float(observed) > max_pde_loss:
                    reasons.append(
                        f"{family}: max normalized PDE loss {float(observed):.6g} "
                        f"> {max_pde_loss:.6g}"
                    )
            if require_validated_solvers:
                loss_statuses = set(dict(row.get("pde_loss_status_counts", {})))
                incomplete_statuses = sorted(loss_statuses - {"measured"})
                if incomplete_statuses:
                    reasons.append(
                        f"{family}: PDE loss is not fully measured: "
                        f"{', '.join(incomplete_statuses)}"
                    )
                fidelities = set(dict(row.get("solver_fidelities", {})))
                unvalidated = sorted(fidelities - VALIDATED_SOLVER_FIDELITIES)
                if unvalidated:
                    reasons.append(
                        f"{family}: unvalidated solver fidelities: {', '.join(unvalidated)}"
                    )
    passed = not reasons
    quality_gate_ready = bool(passed and strict and require_all_pdes and require_validated_solvers)
    return {
        "status": "pass" if passed and strict else "warning" if passed else "fail",
        "strict": bool(strict),
        "max_pde_loss": max_pde_loss,
        "require_all_pdes": bool(require_all_pdes),
        "require_validated_solvers": bool(require_validated_solvers),
        "record_count": record_count,
        "missing_quality_count": missing_quality_count,
        "invalid_quality_count": invalid_quality_count,
        "expected_record_count": expected_record_count,
        "reasons": reasons,
        "quality_gate_ready": quality_gate_ready,
        "publication_ready": False,
        "publication_note": (
            "Canonical full-factor plan coverage, checksum validation, and independently "
            "verified release evidence are required beyond this quality gate."
        ),
    }


def _decode_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    decoded = json.loads(str(value))
    if not isinstance(decoded, dict):
        raise QualityError("HDF5 metadata row is not a JSON object")
    return decoded


def audit_dataset_quality(
    root: str | Path,
    *,
    recompute: bool = False,
    strict: bool = False,
    max_pde_loss: float | None = None,
    require_all_pdes: bool = False,
    require_validated_solvers: bool = False,
) -> dict[str, Any]:
    """Stream all shards and return a dataset-level quality report."""

    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional environment
        raise ModuleNotFoundError("quality auditing requires h5py") from exc

    directory = Path(root)
    shards = sorted({*directory.rglob("*.h5"), *directory.rglob("*.hdf5")})
    if not shards:
        raise QualityError(f"no HDF5 shards found under {directory}")
    accumulator = QualityAccumulator()
    shard_reports: list[dict[str, Any]] = []
    actual_sample_count = 0
    for shard in shards:
        shard_accumulator = QualityAccumulator()
        with h5py.File(shard, "r") as handle:
            required = {"condition", "trajectory", "geometry", "metadata"}
            missing = sorted(required - set(handle.keys()))
            if missing:
                raise QualityError(f"{shard} lacks datasets: {missing}")
            count = int(handle["condition"].shape[0])
            actual_sample_count += count
            for index in range(count):
                metadata = _decode_metadata(handle["metadata"][index])
                quality = metadata.get("quality")
                if recompute:
                    sample = Sample(
                        handle["condition"][index],
                        handle["trajectory"][index],
                        handle["geometry"][index],
                        metadata,
                    )
                    quality = evaluate_sample_quality(sample)
                    metadata = {**metadata, "quality": quality}
                accumulator.update(metadata)
                shard_accumulator.update(metadata)
        shard_reports.append(
            {
                "path": str(shard),
                "samples": count,
                "quality": shard_accumulator.summary(),
            }
        )
    summary = accumulator.summary()
    gate = assess_quality_gate(
        summary,
        strict=strict,
        max_pde_loss=max_pde_loss,
        require_all_pdes=require_all_pdes,
        require_validated_solvers=require_validated_solvers,
        expected_record_count=actual_sample_count,
    )
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "root": str(directory.resolve()),
        "shard_count": len(shards),
        "sample_count": actual_sample_count,
        "quality": summary,
        "gate": gate,
        "shards": shard_reports,
    }


def write_quality_csv(report: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    quality = report.get("quality", report)
    by_pde = quality.get("by_pde", {}) if isinstance(quality, Mapping) else {}
    rows: list[dict[str, Any]] = []
    if isinstance(by_pde, Mapping):
        for family, payload in sorted(by_pde.items()):
            if not isinstance(payload, Mapping):
                continue
            row: dict[str, Any] = {
                "row_type": "pde_summary",
                "pde": family,
                "status": payload.get("status"),
                "sample_count": payload.get("sample_count", 0),
                "operators": " | ".join(payload.get("operators", [])),
            }
            metrics = payload.get("metrics", {})
            if isinstance(metrics, Mapping):
                for metric, stats in metrics.items():
                    if isinstance(stats, Mapping):
                        for name in ("count", "mean", "std", "min", "max", "max_sample_id"):
                            row[f"{metric}.{name}"] = stats.get(name)
            rows.append(row)
    by_calibration_key = (
        quality.get("by_calibration_key", {}) if isinstance(quality, Mapping) else {}
    )
    if isinstance(by_calibration_key, Mapping):
        for calibration_key, payload in sorted(by_calibration_key.items()):
            if not isinstance(payload, Mapping):
                continue
            context = payload.get("calibration_context", {})
            row = {
                "row_type": "calibration_stratum",
                "calibration_key": calibration_key,
                "sample_count": payload.get("sample_count", 0),
                "pde_loss_normalized_max": payload.get("pde_loss_normalized_max"),
            }
            if isinstance(context, Mapping):
                for name, value in context.items():
                    row[f"context.{name}"] = (
                        json.dumps(value, sort_keys=True, separators=(",", ":"))
                        if isinstance(value, (Mapping, list))
                        else value
                    )
            stats = payload.get("pde_loss_normalized")
            if isinstance(stats, Mapping):
                for name in ("count", "mean", "std", "min", "max", "max_sample_id"):
                    row[f"pde_loss_normalized.{name}"] = stats.get(name)
            rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    return destination


__all__ = [
    "BUILTIN_PDE_FAMILIES",
    "QUALITY_SCHEMA_VERSION",
    "QualityAccumulator",
    "QualityError",
    "QualityGateError",
    "assess_quality_gate",
    "audit_dataset_quality",
    "calibration_key_for_context",
    "enforce_generation_quality",
    "evaluate_sample_quality",
    "generation_quality_rejected",
    "normalize_quality_config",
    "summarize_quality_records",
    "write_quality_csv",
]
