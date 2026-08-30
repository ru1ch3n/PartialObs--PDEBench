# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Official partial-observation masks.

All functions return ``bool[H,W]`` with ``True`` denoting an observed value.
The canonical 128x128 3% protocol uses exactly 500 sensors as specified by the
benchmark; callers can request any other exact count through ``count=``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .registry import MASK_REGISTRY
from .schema import normalize_resolution

MASK_PROTOCOL_NAMES = (
    "random_1pct",
    "random_3pct",
    "random_5pct",
    "random_10pct",
    "regular_grid",
    "block_missing",
    "line_sensors",
    "boundary_sensors",
    "clustered_sensors",
)

MAIN_TRAIN_COUNT_128 = 500


def _count_from_ratio(shape: tuple[int, int], ratio: float) -> int:
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("ratio must lie in (0, 1]")
    return min(shape[0] * shape[1], max(1, int(round(np.prod(shape) * ratio))))


def exact_random_mask(
    shape: int | tuple[int, int], count: int, seed: int | np.random.Generator = 0
) -> np.ndarray:
    """Sample exactly ``count`` distinct pixels without touching global RNG state."""

    spatial = normalize_resolution(shape)
    total = spatial[0] * spatial[1]
    if not 0 <= int(count) <= total:
        raise ValueError(f"count must be between 0 and {total}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    mask = np.zeros(total, dtype=bool)
    if count:
        mask[rng.choice(total, size=int(count), replace=False)] = True
    return mask.reshape(spatial)


def _random_protocol(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    default_ratio: float,
    ratio: float | None = None,
    count: int | None = None,
) -> np.ndarray:
    selected_ratio = default_ratio if ratio is None else ratio
    selected = int(count) if count is not None else _count_from_ratio(shape, selected_ratio)
    return exact_random_mask(shape, selected, rng)


@MASK_REGISTRY.register("random_1pct", aliases=("random1", "random_1", "1pct"))
def random_1pct(shape: tuple[int, int], rng: np.random.Generator, **kwargs: Any) -> np.ndarray:
    return _random_protocol(shape, rng, default_ratio=0.01, **kwargs)


@MASK_REGISTRY.register("random_3pct", aliases=("random", "train", "random3", "random_3", "3pct"))
def random_3pct(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    ratio: float | None = None,
    count: int | None = None,
    **_: Any,
) -> np.ndarray:
    if ratio is None and count is None and shape == (128, 128):
        count = MAIN_TRAIN_COUNT_128
    return _random_protocol(shape, rng, default_ratio=0.03, ratio=ratio, count=count)


@MASK_REGISTRY.register("random_5pct", aliases=("random5", "random_5", "5pct"))
def random_5pct(shape: tuple[int, int], rng: np.random.Generator, **kwargs: Any) -> np.ndarray:
    return _random_protocol(shape, rng, default_ratio=0.05, **kwargs)


@MASK_REGISTRY.register("random_10pct", aliases=("random10", "random_10", "10pct"))
def random_10pct(shape: tuple[int, int], rng: np.random.Generator, **kwargs: Any) -> np.ndarray:
    return _random_protocol(shape, rng, default_ratio=0.10, **kwargs)


@MASK_REGISTRY.register("regular_grid", aliases=("grid", "regular-grid"))
def regular_grid(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    ratio: float = 0.03,
    spacing: int | None = None,
    random_offset: bool = False,
    **_: Any,
) -> np.ndarray:
    step = int(spacing or max(1, round(1.0 / np.sqrt(ratio))))
    if step < 1:
        raise ValueError("spacing must be positive")
    offset_y = int(rng.integers(step)) if random_offset else step // 2
    offset_x = int(rng.integers(step)) if random_offset else step // 2
    mask = np.zeros(shape, dtype=bool)
    mask[offset_y::step, offset_x::step] = True
    if not mask.any():
        mask[shape[0] // 2, shape[1] // 2] = True
    return mask


@MASK_REGISTRY.register("block_missing", aliases=("block", "missing_block", "block-missing"))
def block_missing(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    missing_fraction: float = 0.25,
    block_shape: tuple[int, int] | None = None,
    **_: Any,
) -> np.ndarray:
    if not 0.0 < missing_fraction < 1.0:
        raise ValueError("missing_fraction must lie in (0, 1)")
    if block_shape is None:
        scale = np.sqrt(missing_fraction)
        block_height = max(1, min(shape[0], int(round(shape[0] * scale))))
        block_width = max(1, min(shape[1], int(round(shape[1] * scale))))
    else:
        block_height, block_width = map(int, block_shape)
        if not (1 <= block_height <= shape[0] and 1 <= block_width <= shape[1]):
            raise ValueError("block_shape must fit inside the field")
    start_y = int(rng.integers(0, shape[0] - block_height + 1))
    start_x = int(rng.integers(0, shape[1] - block_width + 1))
    mask = np.ones(shape, dtype=bool)
    mask[start_y : start_y + block_height, start_x : start_x + block_width] = False
    return mask


@MASK_REGISTRY.register("line_sensors", aliases=("lines", "line", "line-sensors"))
def line_sensors(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    ratio: float = 0.03,
    num_lines: int | None = None,
    orientation: str = "both",
    **_: Any,
) -> np.ndarray:
    if orientation not in {"horizontal", "vertical", "both"}:
        raise ValueError("orientation must be horizontal, vertical, or both")
    target = _count_from_ratio(shape, ratio)
    if num_lines is None:
        representative_length = max(shape) if orientation != "both" else (shape[0] + shape[1]) / 2
        num_lines = max(1, int(round(target / representative_length)))
    mask = np.zeros(shape, dtype=bool)
    horizontal_count = num_lines if orientation == "horizontal" else 0
    vertical_count = num_lines if orientation == "vertical" else 0
    if orientation == "both":
        horizontal_count = (num_lines + 1) // 2
        vertical_count = num_lines // 2
    if horizontal_count > shape[0] or vertical_count > shape[1]:
        raise ValueError("num_lines exceeds the available distinct sensor lines")
    if horizontal_count:
        mask[rng.choice(shape[0], horizontal_count, replace=False), :] = True
    if vertical_count:
        mask[:, rng.choice(shape[1], vertical_count, replace=False)] = True
    return mask


@MASK_REGISTRY.register("boundary_sensors", aliases=("boundary", "edge", "boundary-sensors"))
def boundary_sensors(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    width: int = 1,
    count: int | None = None,
    **_: Any,
) -> np.ndarray:
    del rng
    if not 1 <= int(width) <= min(shape) // 2:
        raise ValueError("width must fit inside the field")
    mask = np.zeros(shape, dtype=bool)
    mask[:width, :] = mask[-width:, :] = True
    mask[:, :width] = mask[:, -width:] = True
    if count is not None:
        candidates = np.flatnonzero(mask)
        selected = np.zeros(mask.size, dtype=bool)
        # Evenly spaced deterministic boundary probes are preferable here to a
        # random subset because this protocol represents installed edge sensors.
        take = min(int(count), len(candidates))
        selected[candidates[np.linspace(0, len(candidates) - 1, take, dtype=int)]] = True
        mask = selected.reshape(shape)
    return mask


@MASK_REGISTRY.register("clustered_sensors", aliases=("clustered", "clusters", "clustered-sensors"))
def clustered_sensors(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    ratio: float = 0.03,
    count: int | None = None,
    clusters: int = 4,
    spread: float = 0.08,
    **_: Any,
) -> np.ndarray:
    if clusters < 1 or spread <= 0:
        raise ValueError("clusters and spread must be positive")
    selected_count = int(count) if count is not None else _count_from_ratio(shape, ratio)
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    weights = np.zeros(shape, dtype=np.float64)
    sigma = spread * min(shape)
    for _cluster in range(int(clusters)):
        center_y = rng.uniform(0, shape[0] - 1)
        center_x = rng.uniform(0, shape[1] - 1)
        weights += np.exp(-((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * sigma**2))
    probabilities = weights.ravel() + np.finfo(float).eps
    probabilities /= probabilities.sum()
    indices = rng.choice(
        mask_size := shape[0] * shape[1], selected_count, replace=False, p=probabilities
    )
    mask = np.zeros(mask_size, dtype=bool)
    mask[indices] = True
    return mask.reshape(shape)


def generate_mask(
    protocol: str,
    shape: int | tuple[int, int],
    seed: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    """Generate one official mask with deterministic local random state."""

    spatial = normalize_resolution(shape)
    function: Callable[..., np.ndarray] = MASK_REGISTRY.get(protocol)
    mask = np.asarray(function(spatial, rng=np.random.default_rng(int(seed)), **kwargs), dtype=bool)
    if mask.shape != spatial:
        raise RuntimeError(f"mask {protocol!r} returned {mask.shape}, expected {spatial}")
    return mask


def broadcast_mask(mask: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Expand an HW mask across optional time/channel axes of ``target``."""

    spatial_mask = np.asarray(mask, dtype=bool)
    array = np.asarray(target)
    if spatial_mask.ndim != 2:
        raise ValueError("mask must be [H,W]")
    if array.ndim == 2 and array.shape == spatial_mask.shape:
        return spatial_mask
    if array.ndim == 3:
        if tuple(array.shape[:2]) == spatial_mask.shape:  # HWC
            return np.broadcast_to(spatial_mask[..., None], array.shape)
        if tuple(array.shape[1:]) == spatial_mask.shape:  # THW
            return np.broadcast_to(spatial_mask[None, ...], array.shape)
    if array.ndim == 4 and tuple(array.shape[1:3]) == spatial_mask.shape:
        return np.broadcast_to(spatial_mask[None, ..., None], array.shape)
    raise ValueError("target must have HW, HWC, THW, or THWC shape matching mask")


def apply_mask(values: np.ndarray, mask: np.ndarray, *, fill_value: float = 0.0) -> np.ndarray:
    expanded = broadcast_mask(mask, values)
    return np.where(expanded, values, fill_value)


def list_masks() -> tuple[str, ...]:
    return MASK_REGISTRY.names()
