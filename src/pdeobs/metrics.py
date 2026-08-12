"""Numerical, spectral, physical, rollout, and OOD benchmark metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Array = Any


def _arrays(prediction: Array, target: Array) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred.shape} != {truth.shape}")
    return pred, truth


def _reduce(values: np.ndarray, reduction: str) -> float | np.ndarray:
    if reduction == "none":
        return values
    if reduction == "mean":
        return float(np.nanmean(values))
    if reduction == "sum":
        return float(np.nansum(values))
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _sample_axes(array: np.ndarray) -> tuple[int, ...]:
    return tuple(range(1, array.ndim)) if array.ndim > 1 else (0,)


def mse(prediction: Array, target: Array, *, reduction: str = "mean") -> float | np.ndarray:
    pred, truth = _arrays(prediction, target)
    if reduction == "none":
        return (pred - truth) ** 2
    value = np.nanmean((pred - truth) ** 2)
    return float(value) if reduction == "mean" else float(np.nansum((pred - truth) ** 2))


def mae(prediction: Array, target: Array, *, reduction: str = "mean") -> float | np.ndarray:
    pred, truth = _arrays(prediction, target)
    if reduction == "none":
        return np.abs(pred - truth)
    value = np.nanmean(np.abs(pred - truth))
    return float(value) if reduction == "mean" else float(np.nansum(np.abs(pred - truth)))


def relative_l2(
    prediction: Array,
    target: Array,
    *,
    epsilon: float = 1e-12,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Per-sample relative L2, averaged across a batch by default."""

    pred, truth = _arrays(prediction, target)
    axes = _sample_axes(pred)
    numerator = np.sqrt(np.nansum((pred - truth) ** 2, axis=axes))
    denominator = np.sqrt(np.nansum(truth**2, axis=axes))
    values = numerator / np.maximum(denominator, epsilon)
    return _reduce(np.atleast_1d(values), reduction)


relative_l2_error = relative_l2


def _spatial_axes(array: np.ndarray, spatial_axes: tuple[int, int] | None) -> tuple[int, int]:
    if spatial_axes is not None:
        return tuple(axis % array.ndim for axis in spatial_axes)  # type: ignore[return-value]
    if array.ndim < 2:
        raise ValueError("Spectral metrics need at least two spatial dimensions")
    # Recognize channel-last HWC/NHWC/BTHWC when C is small.
    if array.ndim == 3 and array.shape[-1] <= 8 and array.shape[-1] < min(array.shape[:2]):
        return 0, 1
    if array.ndim >= 4 and array.shape[-1] <= 8 and array.shape[-1] < min(array.shape[-3:-1]):
        return array.ndim - 3, array.ndim - 2
    return array.ndim - 2, array.ndim - 1


def _radial_frequency(shape: tuple[int, ...], axes: tuple[int, int]) -> np.ndarray:
    fy = np.fft.fftfreq(shape[axes[0]])
    fx = np.fft.fftfreq(shape[axes[1]])
    radius_2d = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    view = [1] * len(shape)
    view[axes[0]], view[axes[1]] = radius_2d.shape
    return radius_2d.reshape(view)


def frequency_band_errors(
    prediction: Array,
    target: Array,
    *,
    bands: Mapping[str, tuple[float, float]] | None = None,
    spatial_axes: tuple[int, int] | None = None,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Relative Fourier coefficient errors in normalized radial bands.

    Frequencies use cycles per grid cell; the radial Nyquist limit is about
    ``sqrt(0.5)``. Default boundaries are defined as fractions of that limit.
    """

    pred, truth = _arrays(prediction, target)
    axes = _spatial_axes(pred, spatial_axes)
    p_fft = np.fft.fftn(pred, axes=axes, norm="ortho")
    t_fft = np.fft.fftn(truth, axes=axes, norm="ortho")
    radius = _radial_frequency(pred.shape, axes)
    maximum = np.sqrt(0.5)
    definitions = bands or {
        "low": (0.0, maximum / 3.0),
        "mid": (maximum / 3.0, 2.0 * maximum / 3.0),
        "high": (2.0 * maximum / 3.0, maximum + 1e-12),
    }
    output: dict[str, float] = {}
    for name, (lower, upper) in definitions.items():
        selected = (radius >= lower) & (radius < upper)
        numerator = np.sum(np.abs(p_fft - t_fft) ** 2 * selected)
        denominator = np.sum(np.abs(t_fft) ** 2 * selected)
        output[name] = float(np.sqrt(numerator / max(float(denominator), epsilon)))
    return output


spectral_band_error = frequency_band_errors


def spectral_centroid(
    field: Array,
    *,
    spatial_axes: tuple[int, int] | None = None,
    epsilon: float = 1e-12,
) -> float:
    values = np.asarray(field, dtype=np.float64)
    axes = _spatial_axes(values, spatial_axes)
    energy = np.abs(np.fft.fftn(values, axes=axes, norm="ortho")) ** 2
    radius = _radial_frequency(values.shape, axes)
    return float(np.sum(radius * energy) / max(float(np.sum(energy)), epsilon))


def spectral_centroid_error(
    prediction: Array,
    target: Array,
    *,
    spatial_axes: tuple[int, int] | None = None,
) -> float:
    return abs(
        spectral_centroid(prediction, spatial_axes=spatial_axes)
        - spectral_centroid(target, spatial_axes=spatial_axes)
    )


def high_frequency_energy(
    field: Array,
    *,
    cutoff: float | None = None,
    spatial_axes: tuple[int, int] | None = None,
    relative: bool = True,
    epsilon: float = 1e-12,
) -> float:
    values = np.asarray(field, dtype=np.float64)
    axes = _spatial_axes(values, spatial_axes)
    energy = np.abs(np.fft.fftn(values, axes=axes, norm="ortho")) ** 2
    radius = _radial_frequency(values.shape, axes)
    threshold = cutoff if cutoff is not None else 2.0 * np.sqrt(0.5) / 3.0
    high = float(np.sum(energy * (radius >= threshold)))
    return high / max(float(np.sum(energy)), epsilon) if relative else high


def high_frequency_energy_error(
    prediction: Array,
    target: Array,
    *,
    cutoff: float | None = None,
    spatial_axes: tuple[int, int] | None = None,
    epsilon: float = 1e-12,
) -> float:
    predicted = high_frequency_energy(
        prediction, cutoff=cutoff, spatial_axes=spatial_axes, relative=False
    )
    expected = high_frequency_energy(
        target, cutoff=cutoff, spatial_axes=spatial_axes, relative=False
    )
    return float(abs(predicted - expected) / max(abs(expected), epsilon))


def rollout_horizon_metrics(
    prediction: Array,
    target: Array,
    *,
    horizons: Iterable[int] = (1, 2, 4, 8),
    time_axis: int = 1,
) -> dict[str, float]:
    pred, truth = _arrays(prediction, target)
    axis = time_axis % pred.ndim
    steps = pred.shape[axis]
    output: dict[str, float] = {}
    for horizon in sorted(set(int(h) for h in horizons)):
        if horizon < 1 or horizon > steps:
            continue
        index = [slice(None)] * pred.ndim
        index[axis] = horizon - 1
        output[f"rel_l2_h{horizon}"] = float(relative_l2(pred[tuple(index)], truth[tuple(index)]))
        output[f"mse_h{horizon}"] = float(mse(pred[tuple(index)], truth[tuple(index)]))
    output["rel_l2_rollout"] = float(relative_l2(pred, truth))
    return output


def rollout_error_at_horizon(
    prediction: Array,
    target: Array,
    horizon: int,
    *,
    time_axis: int = 1,
) -> float:
    result = rollout_horizon_metrics(prediction, target, horizons=(horizon,), time_axis=time_axis)
    key = f"rel_l2_h{horizon}"
    if key not in result:
        raise ValueError(f"horizon {horizon} exceeds the available rollout")
    return result[key]


def _masked_mean(values: np.ndarray, valid_mask: Array | None) -> float:
    if valid_mask is None:
        return float(np.nanmean(values))
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if values.ndim >= 3 and mask.shape[-2:] == values.shape[-3:-1]:
        mask = mask[..., None]
    while mask.ndim < values.ndim:
        mask = np.expand_dims(mask, axis=1)
    try:
        mask = np.broadcast_to(mask, values.shape)
    except ValueError as exc:
        raise ValueError(
            f"valid_mask shape {np.asarray(valid_mask).shape} cannot mask {values.shape}"
        ) from exc
    if not np.any(mask):
        raise ValueError("valid_mask excludes every value")
    return float(np.nanmean(np.where(mask, values, np.nan)))


def kinetic_energy(
    velocity: Array, *, channel_axis: int = -1, valid_mask: Array | None = None
) -> float:
    values = np.asarray(velocity, dtype=np.float64)
    axis = channel_axis % values.ndim
    if values.shape[axis] < 2:
        raise ValueError("Kinetic energy requires at least two velocity channels")
    components = np.take(values, (0, 1), axis=axis)
    density = 0.5 * np.sum(components**2, axis=axis)
    return _masked_mean(density, valid_mask)


energy = kinetic_energy


def energy_error(
    prediction: Array,
    target: Array,
    *,
    channel_axis: int = -1,
    relative: bool = True,
    epsilon: float = 1e-12,
) -> float:
    predicted = kinetic_energy(prediction, channel_axis=channel_axis)
    expected = kinetic_energy(target, channel_axis=channel_axis)
    difference = abs(predicted - expected)
    return float(difference / max(abs(expected), epsilon) if relative else difference)


def vorticity(
    velocity: Array,
    *,
    channel_axis: int = -1,
    spatial_axes: tuple[int, int] | None = None,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Return 2-D scalar vorticity ``dv/dx - du/dy``."""

    values = np.asarray(velocity, dtype=np.float64)
    channel = channel_axis % values.ndim
    axes = _spatial_axes(values, spatial_axes)
    u = np.take(values, 0, axis=channel)
    v = np.take(values, 1, axis=channel)
    # Removing a channel axis shifts spatial axis numbers above it.
    adjusted = tuple(axis - (axis > channel) for axis in axes)
    du_dy = np.gradient(u, spacing[0], axis=adjusted[0])
    dv_dx = np.gradient(v, spacing[1], axis=adjusted[1])
    return dv_dx - du_dy


def velocity_from_vorticity(
    field: Array,
    *,
    channel_axis: int = -1,
    spatial_axes: tuple[int, int] | None = None,
) -> np.ndarray:
    """Recover periodic divergence-free velocity from scalar vorticity.

    Canonical ``BHWC`` and ``BTHWC`` inputs may retain a singleton vorticity
    channel.  Leading batch/time axes are preserved and the returned velocity
    uses channels-last ``[..., H, W, 2]`` layout on the unit periodic domain.
    """

    values = np.asarray(field, dtype=np.float64)
    axes = _spatial_axes(values, spatial_axes)
    channel = channel_axis % values.ndim
    if channel not in axes:
        if values.shape[channel] != 1:
            raise ValueError("Vorticity input must have one channel")
        omega = np.take(values, 0, axis=channel)
        adjusted_axes = tuple(axis - (axis > channel) for axis in axes)
    else:
        # Plain HW/BHW scalar arrays have no explicit channel dimension.
        omega = values
        adjusted_axes = axes

    ordered = np.moveaxis(omega, adjusted_axes, (-2, -1))
    height, width = ordered.shape[-2:]
    dx, dy = 1.0 / width, 1.0 / height
    flattened = ordered.reshape(-1, height, width)
    velocity = np.empty((*ordered.shape[:-2], height, width, 2), dtype=np.float64)

    from .pdes.common import stream_velocity

    for index, frame in enumerate(flattened):
        velocity_x, velocity_y = stream_velocity(frame, dx, dy)
        velocity.reshape(-1, height, width, 2)[index] = np.stack((velocity_x, velocity_y), axis=-1)
    return velocity


def enstrophy(
    field: Array,
    *,
    is_vorticity: bool = True,
    channel_axis: int = -1,
    spatial_axes: tuple[int, int] | None = None,
    valid_mask: Array | None = None,
) -> float:
    omega = (
        np.asarray(field, dtype=np.float64)
        if is_vorticity
        else vorticity(field, channel_axis=channel_axis, spatial_axes=spatial_axes)
    )
    return 0.5 * _masked_mean(omega**2, valid_mask)


def enstrophy_error(
    prediction: Array,
    target: Array,
    *,
    is_vorticity: bool = True,
    channel_axis: int = -1,
    spatial_axes: tuple[int, int] | None = None,
    relative: bool = True,
    epsilon: float = 1e-12,
) -> float:
    predicted = enstrophy(
        prediction, is_vorticity=is_vorticity, channel_axis=channel_axis, spatial_axes=spatial_axes
    )
    expected = enstrophy(
        target, is_vorticity=is_vorticity, channel_axis=channel_axis, spatial_axes=spatial_axes
    )
    difference = abs(predicted - expected)
    return float(difference / max(abs(expected), epsilon) if relative else difference)


def vorticity_error(
    prediction: Array,
    target: Array,
    *,
    inputs_are_velocity: bool = False,
    channel_axis: int = -1,
    spatial_axes: tuple[int, int] | None = None,
) -> float:
    pred = (
        vorticity(prediction, channel_axis=channel_axis, spatial_axes=spatial_axes)
        if inputs_are_velocity
        else prediction
    )
    truth = (
        vorticity(target, channel_axis=channel_axis, spatial_axes=spatial_axes)
        if inputs_are_velocity
        else target
    )
    return float(relative_l2(pred, truth))


def physical_errors(
    prediction: Array,
    target: Array,
    *,
    representation: str = "vorticity",
    channel_axis: int = -1,
    spatial_axes: tuple[int, int] | None = None,
    valid_mask: Array | None = None,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    pred, truth = _arrays(prediction, target)
    if representation == "velocity":
        p_energy, t_energy = (
            kinetic_energy(pred, channel_axis=channel_axis, valid_mask=valid_mask),
            kinetic_energy(truth, channel_axis=channel_axis, valid_mask=valid_mask),
        )
        p_omega = vorticity(pred, channel_axis=channel_axis, spatial_axes=spatial_axes)
        t_omega = vorticity(truth, channel_axis=channel_axis, spatial_axes=spatial_axes)
    elif representation == "vorticity":
        p_velocity = velocity_from_vorticity(
            pred, channel_axis=channel_axis, spatial_axes=spatial_axes
        )
        t_velocity = velocity_from_vorticity(
            truth, channel_axis=channel_axis, spatial_axes=spatial_axes
        )
        p_energy, t_energy = (
            kinetic_energy(p_velocity, valid_mask=valid_mask),
            kinetic_energy(t_velocity, valid_mask=valid_mask),
        )
        p_omega, t_omega = pred, truth
    else:
        raise ValueError("representation must be 'velocity' or 'vorticity'")
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.ndim and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if p_omega.ndim >= 3 and mask.shape[-2:] == p_omega.shape[-3:-1]:
            mask = mask[..., None]
        while mask.ndim < p_omega.ndim:
            mask = np.expand_dims(mask, axis=1)
        mask = np.broadcast_to(mask, p_omega.shape)
        p_error = np.where(mask, p_omega, np.nan)
        t_error = np.where(mask, t_omega, np.nan)
    else:
        p_error, t_error = p_omega, t_omega
    p_enstrophy = enstrophy(p_omega, valid_mask=valid_mask)
    t_enstrophy = enstrophy(t_omega, valid_mask=valid_mask)
    result = {
        "vorticity_rel_l2": float(relative_l2(p_error, t_error)),
        "enstrophy_relative_error": abs(p_enstrophy - t_enstrophy) / max(abs(t_enstrophy), epsilon),
        "energy_relative_error": abs(p_energy - t_energy) / max(abs(t_energy), epsilon),
    }
    return {key: float(value) for key, value in result.items()}


def ood_degradation(
    iid_score: float,
    ood_score: float,
    *,
    higher_is_better: bool = False,
    mode: str = "ratio",
    epsilon: float = 1e-12,
) -> float:
    """OOD degradation as an error ratio or oriented score difference."""

    if mode == "ratio":
        if higher_is_better:
            return float(iid_score / max(abs(ood_score), epsilon))
        return float(ood_score / max(abs(iid_score), epsilon))
    if mode == "difference":
        return float(iid_score - ood_score if higher_is_better else ood_score - iid_score)
    raise ValueError("mode must be 'ratio' or 'difference'")


def stability_metrics(
    rollout: Array,
    *,
    reference: Array | None = None,
    time_axis: int = 1,
    growth_threshold: float = 10.0,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Finite-value failure rate and trajectory norm-growth diagnostics."""

    values = np.asarray(rollout, dtype=np.float64)
    axis = time_axis % values.ndim
    ordered = np.moveaxis(values, axis, 1)
    flat = (
        ordered.reshape(ordered.shape[0], ordered.shape[1], -1)
        if ordered.ndim > 2
        else ordered[None]
    )
    norms = np.linalg.norm(np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0), axis=-1)
    initial = np.maximum(norms[:, :1], epsilon)
    growth = norms / initial
    invalid = ~np.isfinite(flat).all(axis=(1, 2))
    exploding = np.nanmax(growth, axis=1) > growth_threshold
    result = {
        "nonfinite_rate": float(np.mean(invalid)),
        "growth_failure_rate": float(np.mean(exploding)),
        "stability_failure_rate": float(np.mean(invalid | exploding)),
        "max_norm_growth": float(np.nanmax(growth)),
    }
    if reference is not None:
        result["rollout_rel_l2"] = float(relative_l2(values, reference))
    return result


rollout_stability = stability_metrics
ood_degradation_ratio = ood_degradation


DEFAULT_METRICS: Mapping[str, Callable[[Array, Array], float]] = {
    "relative_l2": relative_l2,
    "mse": mse,
    "mae": mae,
    "spectral_centroid_error": spectral_centroid_error,
    "high_frequency_energy_error": high_frequency_energy_error,
}


@dataclass
class MetricSuite:
    """Extensible callable metric collection used by evaluation runners."""

    metrics: Mapping[str, Callable[[Array, Array], float]] = field(
        default_factory=lambda: dict(DEFAULT_METRICS)
    )
    include_frequency_bands: bool = True

    def __call__(self, prediction: Array, target: Array) -> dict[str, float]:
        result = {name: float(metric(prediction, target)) for name, metric in self.metrics.items()}
        if self.include_frequency_bands:
            result.update(
                {
                    f"spectral_{name}": value
                    for name, value in frequency_band_errors(prediction, target).items()
                }
            )
        return result


BUILTIN_METRICS: Mapping[str, Callable[..., Any]] = {
    **DEFAULT_METRICS,
    "frequency_band_errors": frequency_band_errors,
    "rollout_horizon": rollout_horizon_metrics,
    "energy_error": energy_error,
    "enstrophy_error": enstrophy_error,
    "vorticity_error": vorticity_error,
    "ood_degradation": ood_degradation,
    "stability": stability_metrics,
}


def install_builtin_metrics(registry: Any = None) -> tuple[str, ...]:
    """Expose official metrics through the project registry used by the CLI."""

    if registry is None:
        try:
            from .registry import METRIC_REGISTRY as registry
        except (ImportError, AttributeError):
            return ()
    installed = []
    for name, metric in BUILTIN_METRICS.items():
        if name not in registry:
            registry.register(name, obj=metric)
            installed.append(name)
    return tuple(installed)


install_builtin_metrics()
