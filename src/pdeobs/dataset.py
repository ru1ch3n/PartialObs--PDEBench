"""Task views over immutable canonical PDE-OBS shards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .masks import apply_mask, generate_mask
from .schema import derive_seed
from .storage import LazyHDF5Dataset


def find_shards(root: str | Path, pattern: str = "**/*.h5") -> list[Path]:
    directory = Path(root).expanduser()
    if directory.is_file():
        return [directory]
    return sorted(path for path in directory.glob(pattern) if path.is_file())


class BenchmarkDataset(Sequence[dict[str, Any]]):
    """Expose recovery, forward, inverse, or rollout examples from canonical shards.

    Filtering reads only the small metadata column. Arrays remain lazy and each
    DataLoader worker opens its own HDF5 handle.
    """

    def __init__(
        self,
        shards: str | Path | Sequence[str | Path],
        *,
        task: str = "recovery",
        split: str | None = None,
        filters: Mapping[str, Any] | None = None,
        mask: Mapping[str, Any] | None = None,
        target_step: int = -1,
        horizon: int = 8,
        history_steps: int = 1,
        state_representation: str = "native",
        seed: int = 0,
        max_samples: int | None = None,
        verify: bool = False,
    ) -> None:
        self.base = LazyHDF5Dataset(shards, verify=verify)
        self.task = task.lower().replace("-", "_")
        if self.task not in {"recovery", "forward", "inverse", "rollout"}:
            raise ValueError("task must be recovery, forward, inverse, or rollout")
        self.target_step = int(target_step)
        self.horizon = int(horizon)
        self.history_steps = int(history_steps)
        if self.horizon < 1 or self.history_steps < 1:
            raise ValueError("horizon and history_steps must be positive")
        self.state_representation = str(state_representation).lower()
        if self.state_representation not in {"native", "velocity", "vorticity"}:
            raise ValueError("state_representation must be native, velocity, or vorticity")
        self.seed = int(seed)
        self.mask_config = dict(mask or {"protocol": "random_3pct"})
        self.filters = dict(filters or {})
        if split is not None:
            self.filters["split"] = split

        self.indices: list[int] = []
        self.metadata: list[dict[str, Any]] = []
        sample_ids: set[str] = set()
        signatures: set[tuple[Any, ...]] = set()
        for index, metadata in enumerate(self.base.iter_metadata()):
            if all(_matches(metadata.get(key), expected) for key, expected in self.filters.items()):
                sample_id = str(metadata.get("sample_id", ""))
                if sample_id and sample_id in sample_ids:
                    raise ValueError(f"duplicate sample_id across selected shards: {sample_id}")
                if sample_id:
                    sample_ids.add(sample_id)
                signatures.add(
                    (
                        tuple(metadata.get("resolution", ())),
                        metadata.get("state_representation", "scalar"),
                    )
                )
                self.indices.append(index)
                self.metadata.append(metadata)
                if max_samples is not None and len(self.indices) >= int(max_samples):
                    break
        if not self.indices:
            raise ValueError(f"no samples match filters {self.filters}")
        resolutions = {signature[0] for signature in signatures if signature[0]}
        if len(resolutions) > 1:
            raise ValueError(f"selected shards mix spatial resolutions: {sorted(resolutions)}")
        representations = {
            signature[1] for signature in signatures if signature[1] in {"velocity", "vorticity"}
        }
        if self.state_representation == "native" and len(representations) > 1:
            raise ValueError(
                "selected Navier-Stokes shards mix velocity and vorticity; set "
                "data.state_representation to one canonical representation"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def _mask(self, target: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
        options = dict(self.mask_config)
        protocol = str(options.pop("protocol", "random_3pct"))
        mask_seed = derive_seed(
            self.seed,
            "mask",
            metadata.get("sample_id", metadata.get("seed", 0)),
            protocol,
        )
        spatial = target.shape[:2] if target.ndim <= 3 else target.shape[-3:-1]
        return generate_mask(protocol, spatial, seed=mask_seed, **options)

    def _state_view(
        self,
        condition: np.ndarray,
        trajectory: np.ndarray,
        metadata: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        if metadata.get("pde") != "navier_stokes" or self.state_representation == "native":
            return condition, trajectory
        native = str(metadata.get("state_representation", "native"))
        if native == self.state_representation:
            return condition, trajectory

        height, width = trajectory.shape[1:3]
        dy, dx = 1.0 / height, 1.0 / width

        def to_vorticity(values: np.ndarray) -> np.ndarray:
            u, v = values[..., 0], values[..., 1]
            return (np.gradient(v, dx, axis=-1) - np.gradient(u, dy, axis=-2))[..., None]

        def to_velocity(values: np.ndarray) -> np.ndarray:
            from .pdes.common import stream_velocity

            frames = values if values.ndim == 4 else values[None]
            converted = []
            for frame in frames:
                velocity_x, velocity_y = stream_velocity(frame[..., 0], dx, dy)
                converted.append(np.stack((velocity_x, velocity_y), axis=-1))
            result = np.stack(converted)
            return result if values.ndim == 4 else result[0]

        if native == "velocity" and self.state_representation == "vorticity":
            converted_condition = to_vorticity(condition)
            converted_trajectory = to_vorticity(trajectory)
        elif native == "vorticity" and self.state_representation == "velocity":
            converted_condition = to_velocity(condition)
            converted_trajectory = to_velocity(trajectory)
        else:
            raise ValueError(
                f"cannot convert Navier-Stokes representation {native!r} to "
                f"{self.state_representation!r}"
            )
        metadata["native_state_representation"] = native
        metadata["state_representation"] = self.state_representation
        return converted_condition.astype(np.float32), converted_trajectory.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sample = self.base[self.indices[index]]
        metadata = dict(sample.metadata)
        condition, trajectory = self._state_view(
            np.asarray(sample.condition), np.asarray(sample.trajectory), metadata
        )

        if self.task == "recovery":
            target = np.asarray(trajectory[self.target_step])
            mask = self._mask(target, metadata)
            return {
                "observations": apply_mask(target, mask).astype(np.float32, copy=False),
                "mask": mask[..., None].astype(np.float32),
                "target": target.astype(np.float32, copy=False),
                "geometry": np.asarray(sample.geometry, dtype=np.float32),
                "metadata": metadata,
            }

        if self.task == "forward":
            target = np.asarray(trajectory[self.target_step], dtype=np.float32)
            condition = np.asarray(condition, dtype=np.float32)
            mask = self._mask(condition, metadata)
            return {
                "observations": apply_mask(condition, mask).astype(np.float32, copy=False),
                "mask": mask[..., None].astype(np.float32),
                "target": target,
                "geometry": np.asarray(sample.geometry, dtype=np.float32),
                "metadata": metadata,
            }

        if self.task == "inverse":
            observed_state = np.asarray(trajectory[self.target_step], dtype=np.float32)
            target = np.asarray(condition, dtype=np.float32)
            mask = self._mask(observed_state, metadata)
            return {
                "observations": apply_mask(observed_state, mask).astype(np.float32, copy=False),
                "mask": mask[..., None].astype(np.float32),
                "target": target,
                "geometry": np.asarray(sample.geometry, dtype=np.float32),
                "metadata": metadata,
            }

        required_steps = self.history_steps + self.horizon
        if trajectory.shape[0] < required_steps:
            raise ValueError(
                f"sample {metadata.get('sample_id')} has {trajectory.shape[0]} trajectory "
                f"steps, but rollout requires {self.history_steps} history + "
                f"{self.horizon} future steps"
            )
        history = self.history_steps
        horizon = self.horizon
        history_values = np.asarray(trajectory[:history], dtype=np.float32)
        target = np.asarray(trajectory[history : history + horizon], dtype=np.float32)
        mask = self._mask(history_values, metadata)
        return {
            "observations": apply_mask(history_values, mask).astype(np.float32, copy=False),
            "mask": np.broadcast_to(mask[None, ..., None], (history, *mask.shape, 1)).astype(
                np.float32
            ),
            "target": target,
            "geometry": np.asarray(sample.geometry, dtype=np.float32),
            "metadata": metadata,
        }

    def close(self) -> None:
        self.base.close()

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple, set, frozenset)):
        return value in expected
    return value == expected


def collate_benchmark(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot collate an empty batch")
    keys = {key for row in rows for key in row if key != "metadata"}
    batch: dict[str, Any] = {
        key: np.stack([np.asarray(row[key]) for row in rows]) for key in sorted(keys)
    }
    batch["metadata"] = [dict(row.get("metadata", {})) for row in rows]
    return batch
