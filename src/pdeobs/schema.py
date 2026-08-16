"""Canonical in-memory schema for PDE-OBS generation.

One :class:`Sample` is channel-last ``HWC``/``THWC``.  HDF5 storage adds the
leading sample dimension and is therefore ``NHWC``/``NTHWC``.  Scalar fields may
be passed without their singleton channel and are normalized automatically.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from hashlib import blake2b
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"


def normalize_resolution(resolution: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(resolution, (int, np.integer)):
        height = width = int(resolution)
    else:
        if len(resolution) != 2:
            raise ValueError("resolution must be an integer or (height, width)")
        height, width = (int(resolution[0]), int(resolution[1]))
    if height < 4 or width < 4:
        raise ValueError("each spatial dimension must be at least 4")
    return height, width


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable uint32 seed without relying on randomized ``hash()``."""

    digest = blake2b(digest_size=8, person=b"pdeobs-v1")
    digest.update(str(int(base_seed)).encode("utf-8"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little") % (2**32)


def json_safe(value: Any) -> Any:
    """Convert NumPy/path/dataclass values into stable JSON-compatible data."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _field_hwc(array: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim == 2:
        result = result[..., None]
    if result.ndim != 3:
        raise ValueError(f"{name} must have shape [H,W] or [H,W,C]")
    if result.shape[-1] < 1:
        raise ValueError(f"{name} must contain at least one channel")
    if not np.issubdtype(result.dtype, np.number) and result.dtype != np.bool_:
        raise TypeError(f"{name} must be numeric")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(result)


def _trajectory_thwc(array: np.ndarray, spatial: tuple[int, int]) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim == 2:
        result = result[None, ..., None]
    elif result.ndim == 3:
        # Check THW first: when T=H=W (for example a 9x9 smoke test with
        # nine saved states), both interpretations are otherwise possible.
        if tuple(result.shape[1:]) == spatial:  # scalar THW trajectory
            result = result[..., None]
        elif tuple(result.shape[:2]) == spatial:  # one HWC state
            result = result[None, ...]
        else:
            raise ValueError("trajectory must align with the condition spatial shape")
    if result.ndim != 4:
        raise ValueError("trajectory must have shape [T,H,W] or [T,H,W,C]")
    if tuple(result.shape[1:3]) != spatial:
        raise ValueError("trajectory and condition spatial shapes differ")
    if result.shape[0] < 1 or result.shape[-1] < 1:
        raise ValueError("trajectory needs at least one time and state channel")
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError("trajectory must be numeric")
    if not np.all(np.isfinite(result)):
        raise ValueError("trajectory contains NaN or infinity")
    return np.ascontiguousarray(result)


@dataclass(slots=True)
class Sample:
    """A validated single PDE instance in canonical channel-last form."""

    condition: np.ndarray
    trajectory: np.ndarray
    geometry: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.condition = _field_hwc(self.condition, "condition")
        spatial = tuple(self.condition.shape[:2])
        self.trajectory = _trajectory_thwc(self.trajectory, spatial)
        self.geometry = _field_hwc(self.geometry, "geometry")
        if tuple(self.geometry.shape[:2]) != spatial:
            raise ValueError("geometry and condition spatial shapes differ")
        self.metadata = dict(json_safe(self.metadata))

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return int(self.condition.shape[0]), int(self.condition.shape[1])

    @property
    def time_steps(self) -> int:
        return int(self.trajectory.shape[0])

    @property
    def is_temporal(self) -> bool:
        return self.time_steps > 1

    def astype(self, dtype: str | np.dtype) -> Sample:
        return Sample(
            self.condition.astype(dtype, copy=False),
            self.trajectory.astype(dtype, copy=False),
            self.geometry.astype(dtype, copy=False),
            self.metadata,
        )


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    """Serializable description of one family/boundary/setting/regime case."""

    pde: str
    boundary: str
    setting: str
    regime: str
    num_samples: int = 2000
    resolution: int | tuple[int, int] = 128
    seed: int = 0
    time_steps: int | None = None
    dtype: str = "float32"
    shard_size: int = 100
    tier: str = "full"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("pde", "boundary", "setting", "regime"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if int(self.num_samples) < 1:
            raise ValueError("num_samples must be positive")
        if int(self.shard_size) < 1:
            raise ValueError("shard_size must be positive")
        tier = str(self.tier).strip().lower()
        if tier not in {"tiny", "debug", "signal", "medium", "full", "custom"}:
            raise ValueError("tier must be tiny, debug, signal, medium, full, or custom")
        object.__setattr__(self, "tier", tier)
        object.__setattr__(self, "resolution", normalize_resolution(self.resolution))
        if self.time_steps is not None and int(self.time_steps) < 1:
            raise ValueError("time_steps must be positive")
        np.dtype(self.dtype)
        object.__setattr__(self, "options", dict(json_safe(self.options)))

    @property
    def family(self) -> str:
        return self.pde

    @property
    def samples(self) -> int:
        return int(self.num_samples)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return normalize_resolution(self.resolution)

    @property
    def case_id(self) -> str:
        return "/".join((self.pde, self.boundary, self.setting, self.regime))

    def sample_seed(self, sample_index: int) -> int:
        if sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        return derive_seed(self.seed, self.case_id, int(sample_index))

    def to_dict(self) -> dict[str, Any]:
        return dict(json_safe(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GenerationSpec:
        values = dict(data)
        if "pde" not in values and "family" in values:
            values["pde"] = values.pop("family")
        if "num_samples" not in values:
            for alias in ("samples", "count", "n_samples"):
                if alias in values:
                    values["num_samples"] = values.pop(alias)
                    break
        if isinstance(values.get("resolution"), list):
            values["resolution"] = tuple(values["resolution"])
        return cls(**values)
