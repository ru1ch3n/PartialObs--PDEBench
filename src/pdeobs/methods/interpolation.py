# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Transparent, non-learning partial-observation baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import MethodCapabilities, register_method

_RECOVERY = MethodCapabilities(
    tasks=frozenset({"recovery", "forward", "inverse"}), requires_mask=True
)


def _as_bhwc(values: Any, mask: Any | None) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Canonicalise 2-D fields to [batch, height, width, channels]."""

    x = np.asarray(values)
    original = x.shape
    if x.ndim == 2:
        x = x[None, ..., None]
    elif x.ndim == 3:
        # A mask with the same first two axes indicates HWC; otherwise NHW.
        mshape = np.shape(mask) if mask is not None else ()
        if mshape[:2] == x.shape[:2] and (len(mshape) == 2 or mshape[-1] in (1, x.shape[-1])):
            x = x[None, ...]
        else:
            x = x[..., None]
    elif x.ndim != 4:
        raise ValueError("Expected a 2-D field with shape HW, HWC, NHW, or NHWC")

    if mask is None:
        m = np.isfinite(x)
    else:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 2:
            m = m[None, ..., None]
        elif m.ndim == 3:
            if m.shape == x.shape[:3]:
                m = m[..., None]
            elif x.shape[0] == 1 and m.shape[:2] == x.shape[1:3]:
                m = m[None, ...]
        if m.ndim != 4:
            raise ValueError("Mask is not broadcastable to the observation field")
        try:
            m = np.broadcast_to(m, x.shape)
        except ValueError as exc:
            raise ValueError(f"Mask shape {m.shape} is not broadcastable to {x.shape}") from exc
    return np.asarray(x), np.asarray(m), original


def _restore(x: np.ndarray, original: tuple[int, ...]) -> np.ndarray:
    if len(original) == 2:
        return x[0, ..., 0]
    if len(original) == 3:
        # This exactly distinguishes NHW from HWC only when the canonical batch
        # equals the original first axis.
        return x[..., 0] if x.shape[0] == original[0] else x[0]
    return x


class _ArrayMethod:
    capabilities = _RECOVERY

    def __call__(self, observations: Any, mask: Any | None = None, **kwargs: Any) -> np.ndarray:
        return self.predict(observations, mask, **kwargs)


@register_method("zero", aliases=("zero_fill",))
@dataclass
class ZeroFill(_ArrayMethod):
    """Keep observed entries and fill missing entries with zero."""

    fill_value: float = 0.0
    name: str = "zero"

    def predict(self, observations: Any, mask: Any | None = None, **_: Any) -> np.ndarray:
        x, m, original = _as_bhwc(observations, mask)
        out = np.where(m, np.nan_to_num(x, nan=self.fill_value), self.fill_value)
        return _restore(out, original)


@register_method("mean", aliases=("mean_fill",))
@dataclass
class MeanFill(_ArrayMethod):
    """Fill each sample/channel with its observed spatial mean."""

    empty_value: float = 0.0
    name: str = "mean"

    def predict(self, observations: Any, mask: Any | None = None, **_: Any) -> np.ndarray:
        x, m, original = _as_bhwc(observations, mask)
        valid = m & np.isfinite(x)
        count = valid.sum(axis=(1, 2), keepdims=True)
        total = np.where(valid, x, 0.0).sum(axis=(1, 2), keepdims=True)
        means = np.divide(
            total, count, out=np.full_like(total, self.empty_value, dtype=float), where=count > 0
        )
        return _restore(np.where(valid, x, means), original)


def _nearest_single(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    known = np.argwhere(valid)
    missing = np.argwhere(~valid)
    if not len(known):
        out[...] = 0.0
        return out
    if not len(missing):
        return out
    try:
        from scipy.spatial import cKDTree

        _, indices = cKDTree(known).query(missing)
        out[tuple(missing.T)] = out[tuple(known[indices].T)]
    except ImportError:
        # Chunking avoids constructing an H*W by H*W distance matrix.
        for start in range(0, len(missing), 1024):
            query = missing[start : start + 1024]
            distances = ((query[:, None] - known[None, :]) ** 2).sum(axis=-1)
            nearest = known[np.argmin(distances, axis=1)]
            out[tuple(query.T)] = out[tuple(nearest.T)]
    return out


@register_method("nearest", aliases=("nearest_neighbor", "nearest_interpolation"))
@dataclass
class NearestInterpolation(_ArrayMethod):
    name: str = "nearest"

    def predict(self, observations: Any, mask: Any | None = None, **_: Any) -> np.ndarray:
        x, m, original = _as_bhwc(observations, mask)
        out = np.empty_like(x, dtype=float)
        for b in range(x.shape[0]):
            for c in range(x.shape[-1]):
                valid = m[b, :, :, c] & np.isfinite(x[b, :, :, c])
                out[b, :, :, c] = _nearest_single(x[b, :, :, c], valid)
        return _restore(out, original)


def _linear_single(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linear scattered interpolation, with a dependency-free separable fallback."""

    yy, xx = np.indices(values.shape)
    points = np.column_stack((yy[valid], xx[valid]))
    if not len(points):
        return np.zeros_like(values, dtype=float)
    if len(points) < 3:
        return _nearest_single(values, valid)
    try:
        from scipy.interpolate import griddata

        out = griddata(points, values[valid], (yy, xx), method="linear")
        nearest = _nearest_single(values, valid)
        return np.where(np.isfinite(out), out, nearest)
    except ImportError:
        out = np.asarray(values, dtype=float).copy()
        support = valid.copy()
        # Interpolate along rows and then columns. This is exact bilinear-style
        # interpolation for rectilinear sensor grids and a documented heuristic
        # for arbitrary point masks.
        for row in range(out.shape[0]):
            known = np.flatnonzero(support[row])
            if len(known):
                out[row] = np.interp(np.arange(out.shape[1]), known, out[row, known])
                support[row] = True
        for col in range(out.shape[1]):
            known = np.flatnonzero(support[:, col])
            if len(known):
                out[:, col] = np.interp(np.arange(out.shape[0]), known, out[known, col])
                support[:, col] = True
        if not support.all():
            out = _nearest_single(out, support)
        return out


@register_method("bilinear", aliases=("linear_interpolation",))
@dataclass
class BilinearInterpolation(_ArrayMethod):
    """Piecewise-linear interpolation with nearest extrapolation."""

    name: str = "bilinear"

    def predict(self, observations: Any, mask: Any | None = None, **_: Any) -> np.ndarray:
        x, m, original = _as_bhwc(observations, mask)
        out = np.empty_like(x, dtype=float)
        for b in range(x.shape[0]):
            for c in range(x.shape[-1]):
                valid = m[b, :, :, c] & np.isfinite(x[b, :, :, c])
                out[b, :, :, c] = _linear_single(x[b, :, :, c], valid)
        return _restore(out, original)


@register_method("rbf", aliases=("rbf_interpolation",))
@dataclass
class RBFInterpolation(_ArrayMethod):
    """Gaussian radial-basis interpolation with bounded memory use."""

    smoothing: float = 1e-6
    epsilon: float | None = None
    max_centers: int = 512
    name: str = "rbf"

    def _predict_single(self, values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        yy, xx = np.indices(values.shape)
        points = np.column_stack((yy[valid], xx[valid])).astype(float)
        targets = values[valid].astype(float)
        if not len(points):
            return np.zeros_like(values, dtype=float)
        if len(points) == 1:
            return np.full_like(values, targets[0], dtype=float)
        if len(points) > self.max_centers:
            # Evenly spaced deterministic subsampling makes HPC runs repeatable.
            keep = np.linspace(0, len(points) - 1, self.max_centers, dtype=int)
            points, targets = points[keep], targets[keep]
        query = np.column_stack((yy.ravel(), xx.ravel())).astype(float)
        try:
            from scipy.interpolate import RBFInterpolator

            model = RBFInterpolator(
                points,
                targets,
                kernel="gaussian",
                epsilon=self.epsilon or 1.0,
                smoothing=self.smoothing,
            )
            return model(query).reshape(values.shape)
        except (ImportError, np.linalg.LinAlgError):
            scale = self.epsilon
            if scale is None:
                span = np.maximum(np.ptp(points, axis=0), 1.0)
                scale = float(np.linalg.norm(span) / np.sqrt(len(points)))
            scale = max(float(scale), np.finfo(float).eps)
            distance = ((points[:, None] - points[None, :]) ** 2).sum(-1)
            kernel = np.exp(-distance / (2.0 * scale**2))
            kernel.flat[:: len(points) + 1] += self.smoothing
            weights = np.linalg.lstsq(kernel, targets, rcond=1e-8)[0]
            result = np.empty(len(query), dtype=float)
            for start in range(0, len(query), 4096):
                q = query[start : start + 4096]
                d2 = ((q[:, None] - points[None, :]) ** 2).sum(-1)
                result[start : start + len(q)] = np.exp(-d2 / (2.0 * scale**2)) @ weights
            return result.reshape(values.shape)

    def predict(self, observations: Any, mask: Any | None = None, **_: Any) -> np.ndarray:
        x, m, original = _as_bhwc(observations, mask)
        out = np.empty_like(x, dtype=float)
        for b in range(x.shape[0]):
            for c in range(x.shape[-1]):
                valid = m[b, :, :, c] & np.isfinite(x[b, :, :, c])
                out[b, :, :, c] = self._predict_single(x[b, :, :, c], valid)
        return _restore(out, original)


@register_method("persistence", aliases=("last_value",))
@dataclass
class Persistence:
    """Repeat the last observed state for every requested rollout step."""

    name: str = "persistence"
    capabilities = MethodCapabilities(tasks=frozenset({"rollout"}), temporal=True)

    def predict(
        self,
        observations: Any,
        mask: Any | None = None,
        *,
        horizon: int = 1,
        time_axis: int | None = None,
        **_: Any,
    ) -> np.ndarray:
        x = np.asarray(observations)
        if horizon < 1:
            raise ValueError("horizon must be at least one")
        # Temporal batches conventionally use [N,T,H,W,C]. A single trajectory
        # may use [T,H,W,C]. Callers can override by passing an already sliced
        # final state (HWC/NHWC).
        if x.ndim >= 5:
            last = x[:, -1]
            return np.repeat(last[:, None], horizon, axis=1)
        if time_axis is not None:
            axis = time_axis % x.ndim
            last = np.take(x, -1, axis=axis)
            return np.repeat(np.expand_dims(last, axis), horizon, axis=axis)
        axis = 1 if x.ndim >= 4 else 0
        return np.repeat(np.expand_dims(x, axis), horizon, axis=axis)
