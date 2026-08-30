# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Reproducible recovery and rollout training loops.

The adapter accepts ordinary PyTorch ``DataLoader`` batches, including batches
backed by lazily opened HDF5 datasets. No h5py objects are retained here, which
keeps multi-worker loading and Seawulf jobs safe when the dataset opens files in
each worker process.
"""

from __future__ import annotations

import inspect
import json
import os
import random
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - optional for non-learning baselines
    torch = None
    Tensor = Any
    nn = None


@dataclass
class TrainingConfig:
    task: str = "recovery"
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "relative_l2"
    horizon: int = 1
    history_steps: int = 1
    rollout_target_offset: int = 1
    target_step: int = -1
    data_layout: str = "auto"
    device: str = "auto"
    amp: bool = True
    grad_clip: float | None = 1.0
    seed: int = 0
    deterministic: bool = True
    checkpoint_dir: str = "runs/checkpoints"
    checkpoint_every: int = 1
    resume_from: str | None = None
    strict_resume: bool = True
    monitor: str = "val_loss"
    early_stopping_patience: int | None = None
    teacher_forcing_ratio: float = 0.0
    log_every: int = 20

    def __post_init__(self) -> None:
        self.task = self.task.lower().replace("-", "_")
        if self.task not in {"recovery", "forward", "inverse", "rollout"}:
            raise ValueError("task must be recovery, forward, inverse, or rollout")
        if self.epochs < 1 or self.horizon < 1 or self.history_steps < 1:
            raise ValueError("epochs, horizon, and history_steps must be positive")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        if self.rollout_target_offset < 0:
            raise ValueError("rollout_target_offset must be non-negative")
        if self.data_layout not in {"auto", "channels_first", "channels_last"}:
            raise ValueError("data_layout must be auto, channels_first, or channels_last")
        if not 0.0 <= self.teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be between zero and one")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> TrainingConfig:
        """Create a config while rejecting misspelled keys early."""

        known = {item.name for item in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown training configuration keys: {', '.join(sorted(unknown))}")
        return cls(**dict(values))


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "Training neural baselines requires PyTorch. Install pdeobs[torch] or torch."
        )


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:  # older PyTorch
                torch.use_deterministic_algorithms(True)


def _numpy_rng_payload(state: tuple[Any, ...]) -> dict[str, Any]:
    """Convert NumPy's RNG tuple to weights-only-safe primitive values."""

    return {
        "bit_generator": str(state[0]),
        "keys": np.asarray(state[1], dtype=np.uint32).tolist(),
        "position": int(state[2]),
        "has_gaussian": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _restore_numpy_rng(payload: Mapping[str, Any]) -> None:
    np.random.set_state(
        (
            str(payload["bit_generator"]),
            np.asarray(payload["keys"], dtype=np.uint32),
            int(payload["position"]),
            int(payload["has_gaussian"]),
            float(payload["cached_gaussian"]),
        )
    )


def load_checkpoint_payload(path: str | Path, *, map_location: Any = "cpu") -> Mapping[str, Any]:
    """Load tensor/state dictionaries without enabling arbitrary pickle execution."""

    _require_torch()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Checkpoint {source} is not compatible with safe weights-only loading. "
            "PDE-OBS refuses arbitrary-pickle checkpoints; recreate it with this version "
            "or export a plain state_dict."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint {source} must contain a state mapping")
    return payload


def resolve_device(requested: str = "auto") -> torch.device:
    _require_torch()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        return torch.device(f"cuda:{local_rank % max(torch.cuda.device_count(), 1)}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_tensor(
    value: Any, device: torch.device, layout: str, *, is_mask: bool = False
) -> Tensor | None:
    if value is None:
        return None
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    if tensor.ndim == 2:  # an unbatched scalar field
        tensor = tensor[None, None]
    elif tensor.ndim == 3:  # B,H,W
        tensor = tensor[:, None]
    elif tensor.ndim == 4:
        channel_last = layout == "channels_last" or (
            layout == "auto" and tensor.shape[-1] <= 8 and tensor.shape[-1] < tensor.shape[1]
        )
        if channel_last:  # B,H,W,C -> B,C,H,W
            tensor = tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 5:
        channel_last = layout == "channels_last" or (
            layout == "auto" and tensor.shape[-1] <= 8 and tensor.shape[-1] < tensor.shape[2]
        )
        if channel_last:  # B,T,H,W,C -> B,T,C,H,W
            tensor = tensor.permute(0, 1, 4, 2, 3)
    if is_mask:
        tensor = (tensor > 0).to(dtype=torch.float32)
    return tensor.contiguous()


_INPUT_KEYS = ("observations", "observation", "observed", "y", "input", "inputs", "condition")
_MASK_KEYS = ("mask", "observation_mask", "sensor_mask")
_TARGET_KEYS = ("target", "targets", "trajectory", "solution", "state", "field")


def _first(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any | None:
    return next((mapping[key] for key in keys if key in mapping), None)


def unpack_batch_context(
    batch: Any,
) -> tuple[Any | None, Any | None, Any, Any | None, Any | None]:
    """Return input, mask, target, geometry, and metadata without dropping context."""

    metadata = None
    if isinstance(batch, Mapping):
        inputs = _first(batch, _INPUT_KEYS)
        mask = _first(batch, _MASK_KEYS)
        target = _first(batch, _TARGET_KEYS)
        metadata = batch.get("metadata")
        geometry = batch.get("geometry")
        if target is None:
            raise KeyError(f"Batch needs one target key from {_TARGET_KEYS}")
        return inputs, mask, target, geometry, metadata
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            return batch[0], None, batch[1], None, None
        if len(batch) == 3:
            return batch[0], batch[1], batch[2], None, None
        if len(batch) == 4:
            return batch[0], batch[1], batch[2], None, batch[3]
        if len(batch) == 5:
            return batch[0], batch[1], batch[2], batch[3], batch[4]
    # Canonical LazyHDF5Dataset items are Sample dataclasses. This branch avoids
    # importing schema.py and keeps the training layer usable by external data
    # packages exposing the same attributes.
    if hasattr(batch, "condition") and hasattr(batch, "trajectory"):
        return (
            batch.condition,
            None,
            batch.trajectory,
            getattr(batch, "geometry", None),
            getattr(batch, "metadata", None),
        )
    raise TypeError("Batch must be a mapping or a 2-5 item tuple/list")


def unpack_batch(batch: Any) -> tuple[Any | None, Any | None, Any, Any | None]:
    """Backward-compatible input, mask, target, metadata batch adapter."""

    inputs, mask, target, _, metadata = unpack_batch_context(batch)
    return inputs, mask, target, metadata


def collate_samples(samples: Iterable[Any]) -> dict[str, Any]:
    """Collate canonical HDF5 ``Sample`` objects for a PyTorch DataLoader.

    Use ``DataLoader(dataset, collate_fn=collate_samples)``. Metadata remains a
    list of dictionaries; arrays are stacked without opening or retaining HDF5
    handles in worker processes.
    """

    rows = list(samples)
    if not rows:
        raise ValueError("cannot collate an empty sample list")
    if not all(hasattr(row, "condition") and hasattr(row, "trajectory") for row in rows):
        raise TypeError("collate_samples expects canonical Sample-like objects")
    return {
        "condition": np.stack([np.asarray(row.condition) for row in rows]),
        "trajectory": np.stack([np.asarray(row.trajectory) for row in rows]),
        "geometry": np.stack([np.asarray(row.geometry) for row in rows]),
        "metadata": [getattr(row, "metadata", {}) for row in rows],
    }


def prepare_batch_with_context(
    batch: Any, config: TrainingConfig, device: torch.device
) -> tuple[Tensor, Tensor | None, Tensor, Tensor | None, Any | None]:
    raw_input, raw_mask, raw_target, raw_geometry, metadata = unpack_batch_context(batch)
    target = _to_tensor(raw_target, device, config.data_layout)
    mask = _to_tensor(raw_mask, device, config.data_layout, is_mask=True)
    inputs = _to_tensor(raw_input, device, config.data_layout)
    geometry = _to_tensor(raw_geometry, device, config.data_layout)
    assert target is not None

    if config.task == "rollout":
        if target.ndim != 5:
            raise ValueError("rollout targets must have shape BTCHW (or BTHWC before adaptation)")
        if inputs is None:
            required_steps = config.history_steps + config.horizon
            if target.shape[1] < required_steps:
                raise ValueError(
                    f"rollout trajectory has {target.shape[1]} steps, but requires "
                    f"{config.history_steps} history + {config.horizon} future steps"
                )
            history = config.history_steps
            horizon = config.horizon
            inputs = target[:, :history]
            target = target[:, history : history + horizon]
            if mask is not None and mask.ndim == 5:
                mask = mask[:, :history]
        else:
            # Canonical HDF5 trajectories include t0, while condition stores the
            # initial state. Set rollout_target_offset=0 for loaders whose target
            # tensor already begins at the first future state.
            start = config.rollout_target_offset
            available = target.shape[1] - start
            if available < config.horizon:
                raise ValueError(
                    f"rollout target has {max(0, available)} steps after offset {start}, "
                    f"but horizon {config.horizon} was requested"
                )
            horizon = config.horizon
            target = target[:, start : start + horizon]
    else:
        if target.ndim == 5:
            target = target[:, config.target_step]
        if inputs is None:
            inputs = target
            if mask is not None:
                inputs = inputs * mask
    assert inputs is not None
    return inputs, mask, target, geometry, metadata


def prepare_batch(
    batch: Any, config: TrainingConfig, device: torch.device
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Prepare the three legacy tensors; use prepare_batch_with_context for plugins."""

    inputs, mask, target, _, _ = prepare_batch_with_context(batch, config, device)
    return inputs, mask, target


def _loss_function(name: str):
    key = name.lower().replace("-", "_")
    if key == "mse":
        return lambda prediction, target: torch.mean((prediction - target) ** 2)
    if key == "mae":
        return lambda prediction, target: torch.mean(torch.abs(prediction - target))
    if key in {"relative_l2", "rel_l2"}:

        def relative(prediction: Tensor, target: Tensor) -> Tensor:
            difference = (prediction - target).reshape(prediction.shape[0], -1)
            reference = target.reshape(target.shape[0], -1)
            return torch.mean(
                torch.linalg.vector_norm(difference, dim=1)
                / torch.clamp(torch.linalg.vector_norm(reference, dim=1), min=1e-12)
            )

        return relative
    raise ValueError("loss must be mse, mae, or relative_l2")


def _forward(
    model: nn.Module,
    inputs: Tensor,
    mask: Tensor | None,
    target: Tensor,
    config: TrainingConfig,
    training: bool,
    geometry: Tensor | None = None,
    metadata: Any | None = None,
) -> Tensor:
    def invoke(**kwargs: Any) -> Tensor:
        callable_target = model.forward if hasattr(model, "forward") else model
        try:
            signature = inspect.signature(callable_target)
        except (TypeError, ValueError):
            return model(inputs, **kwargs)
        accepts_extra = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        context = {"geometry": geometry, "metadata": metadata}
        for name, value in context.items():
            if value is not None and (accepts_extra or name in signature.parameters):
                kwargs[name] = value
        filtered = (
            kwargs
            if accepts_extra
            else {name: value for name, value in kwargs.items() if name in signature.parameters}
        )
        return model(inputs, **filtered)

    if config.task == "rollout":
        teacher = target if training and random.random() < config.teacher_forcing_ratio else None
        return invoke(mask=mask, horizon=target.shape[1], teacher_forcing=teacher)
    return invoke(mask=mask)


class Trainer:
    """Stateful trainer with portable checkpoints and resume support."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | Mapping[str, Any],
        optimizer: Any | None = None,
    ) -> None:
        _require_torch()
        self.config = (
            config if isinstance(config, TrainingConfig) else TrainingConfig.from_mapping(config)
        )
        seed_everything(self.config.seed, self.config.deterministic)
        self.device = resolve_device(self.config.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.loss_fn = _loss_function(self.config.loss)
        scaler_enabled = self.config.amp and self.device.type == "cuda"
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=scaler_enabled)
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
            else torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        )
        self.start_epoch = 0
        self.best_metric = float("inf")
        self.history: list[dict[str, float | int]] = []
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)

    @property
    def is_primary(self) -> bool:
        return int(os.environ.get("RANK", "0")) == 0

    def _autocast(self):
        if self.device.type == "cuda":
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast("cuda", enabled=self.config.amp)
            return torch.cuda.amp.autocast(enabled=self.config.amp)
        return nullcontext()

    def run_epoch(self, loader: Iterable[Any], *, training: bool) -> float:
        self.model.train(training)
        total, count = 0.0, 0
        for batch in loader:
            inputs, mask, target, geometry, metadata = prepare_batch_with_context(
                batch, self.config, self.device
            )
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), self._autocast():
                prediction = _forward(
                    self.model,
                    inputs,
                    mask,
                    target,
                    self.config,
                    training,
                    geometry,
                    metadata,
                )
                if prediction.shape != target.shape:
                    raise ValueError(
                        f"Model output {tuple(prediction.shape)} does not match target {tuple(target.shape)}"
                    )
                loss = self.loss_fn(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss encountered: {loss.item()}")
            if training:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            batch_size = target.shape[0]
            total += float(loss.detach()) * batch_size
            count += batch_size
        if not count:
            raise ValueError("Data loader produced no batches")
        return total / count

    def fit(
        self, train_loader: Iterable[Any], val_loader: Iterable[Any] | None = None
    ) -> list[dict[str, float | int]]:
        stale_epochs = 0
        for epoch in range(self.start_epoch, self.config.epochs):
            train_loss = self.run_epoch(train_loader, training=True)
            val_loss = (
                self.run_epoch(val_loader, training=False) if val_loader is not None else train_loss
            )
            row: dict[str, float | int] = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            self.history.append(row)
            improved = val_loss < self.best_metric
            if improved:
                self.best_metric, stale_epochs = val_loss, 0
                if self.is_primary:
                    self.save_checkpoint("best.pt", epoch + 1)
            else:
                stale_epochs += 1
            if self.is_primary and (
                (epoch + 1) % self.config.checkpoint_every == 0 or epoch + 1 == self.config.epochs
            ):
                self.save_checkpoint("last.pt", epoch + 1)
            if (
                self.config.early_stopping_patience is not None
                and stale_epochs >= self.config.early_stopping_patience
            ):
                break
        return self.history

    def _model_state(self) -> Mapping[str, Tensor]:
        return (
            self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict()
        )

    def save_checkpoint(self, filename: str, epoch: int) -> Path:
        directory = Path(self.config.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        payload = {
            "format_version": 1,
            "epoch": epoch,
            "model_state": self._model_state(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "history": self.history,
            "config": asdict(self.config),
            "rng": {
                "python": random.getstate(),
                "numpy": _numpy_rng_payload(np.random.get_state()),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        (directory / "training_config.json").write_text(
            json.dumps(asdict(self.config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination

    def load_checkpoint(self, path: str | Path) -> Mapping[str, Any]:
        checkpoint = load_checkpoint_payload(path, map_location=self.device)
        target = self.model.module if hasattr(self.model, "module") else self.model
        target.load_state_dict(checkpoint["model_state"], strict=self.config.strict_resume)
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.best_metric = float(checkpoint.get("best_metric", float("inf")))
        self.history = list(checkpoint.get("history", []))
        rng = checkpoint.get("rng", {})
        if rng:
            random.setstate(rng["python"])
            _restore_numpy_rng(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        return checkpoint


def train_model(
    model: nn.Module,
    train_loader: Iterable[Any],
    val_loader: Iterable[Any] | None = None,
    *,
    config: TrainingConfig | Mapping[str, Any] | None = None,
) -> Trainer:
    """Convenience API used by the CLI and Python examples."""

    trainer = Trainer(model, config or TrainingConfig())
    trainer.fit(train_loader, val_loader)
    return trainer
