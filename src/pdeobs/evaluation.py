"""Inference and evaluation runners for learning and non-learning methods."""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .metrics import (
    MetricSuite,
    ood_degradation,
    physical_errors,
    rollout_horizon_metrics,
    stability_metrics,
)
from .training import (
    TrainingConfig,
    _forward,
    prepare_batch_with_context,
    resolve_device,
    unpack_batch_context,
)

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class EvaluationConfig:
    task: str = "recovery"
    horizon: int = 8
    history_steps: int = 1
    rollout_target_offset: int = 1
    target_step: int = -1
    data_layout: str = "auto"
    device: str = "auto"
    horizons: tuple[int, ...] = (1, 2, 4, 8)
    include_frequency_bands: bool = True
    include_stability: bool = True
    ood_views: tuple[str, ...] = ()
    mask_protocols: tuple[str, ...] = ()
    physical_representation: str | None = None
    # Evaluation normalizes predictions to channels-last arrays (BHWC/BTHWC).
    channel_axis: int = -1
    spatial_axes: tuple[int, int] = (-3, -2)
    predictions_path: str | None = None
    report_path: str | None = None

    def training_adapter(self) -> TrainingConfig:
        return TrainingConfig(
            task=self.task,
            horizon=self.horizon,
            history_steps=self.history_steps,
            rollout_target_offset=self.rollout_target_offset,
            target_step=self.target_step,
            data_layout=self.data_layout,
            device=self.device,
            epochs=1,
            amp=False,
        )


def _is_torch_model(method: Any) -> bool:
    return torch is not None and nn is not None and isinstance(method, nn.Module)


def _numpy_channels_last(value: Any, layout: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 4:
        channel_first = layout == "channels_first" or (
            layout == "auto" and array.shape[1] <= 8 and array.shape[1] < array.shape[-1]
        )
        if channel_first:
            array = np.moveaxis(array, 1, -1)
    elif array.ndim == 5:
        channel_first = layout == "channels_first" or (
            layout == "auto" and array.shape[2] <= 8 and array.shape[2] < array.shape[-1]
        )
        if channel_first:
            array = np.moveaxis(array, 2, -1)
    return array


def _prepare_numpy_batch(
    batch: Any, config: EvaluationConfig
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, Any | None]:
    raw_input, raw_mask, raw_target, raw_geometry, metadata = unpack_batch_context(batch)
    target = _numpy_channels_last(raw_target, config.data_layout)
    inputs = None if raw_input is None else _numpy_channels_last(raw_input, config.data_layout)
    mask = (
        None
        if raw_mask is None
        else _numpy_channels_last(raw_mask, config.data_layout).astype(bool)
    )
    geometry = (
        None if raw_geometry is None else _numpy_channels_last(raw_geometry, config.data_layout)
    )
    if config.task == "rollout":
        if target.ndim != 5:
            raise ValueError("rollout targets must have shape BTHWC or BTCHW")
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
            inputs = target if mask is None else np.where(mask, target, 0.0)
    assert inputs is not None
    return inputs, mask, target, geometry, metadata


def predict_batch(
    method: Any, batch: Any, config: EvaluationConfig, device: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    if _is_torch_model(method):
        if device is None:
            device = resolve_device(config.device)
        adapter = config.training_adapter()
        inputs, mask, target, geometry, metadata = prepare_batch_with_context(
            batch, adapter, device
        )
        method = method.to(device)
        method.eval()
        with torch.no_grad():
            prediction = _forward(
                method,
                inputs,
                mask,
                target,
                adapter,
                False,
                geometry,
                metadata,
            )
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {tuple(prediction.shape)} differs from target {tuple(target.shape)}"
            )
        # Training uses channels-first tensors.  Evaluation has one canonical
        # representation so physical and spectral axes do not depend on which
        # method produced the prediction.
        predicted = _numpy_channels_last(prediction.detach().cpu().numpy(), "channels_first")
        expected = _numpy_channels_last(target.detach().cpu().numpy(), "channels_first")
        return predicted, expected
    else:
        observations, observation_mask, target, geometry, metadata = _prepare_numpy_batch(
            batch, config
        )
        kwargs = {"horizon": target.shape[1]} if config.task == "rollout" else {}
        callable_target = method.predict if hasattr(method, "predict") else method
        if callable(callable_target):
            try:
                signature = inspect.signature(callable_target)
            except (TypeError, ValueError):
                signature = None
            context = {"geometry": geometry, "metadata": metadata}
            if signature is not None:
                accepts_extra = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
                for name, value in context.items():
                    if value is not None and (accepts_extra or name in signature.parameters):
                        kwargs[name] = value
                if not accepts_extra:
                    kwargs = {
                        name: value
                        for name, value in kwargs.items()
                        if name in signature.parameters
                    }
            result = callable_target(observations, observation_mask, **kwargs)
        else:
            raise TypeError("method must be a PyTorch module, expose predict(), or be callable")
        prediction = np.asarray(result)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} differs from target {target.shape}"
            )
        return prediction, target


def evaluate_predictions(
    prediction: Any,
    target: Any,
    config: EvaluationConfig | None = None,
    *,
    geometry: Any | None = None,
) -> dict[str, float]:
    config = config or EvaluationConfig()
    suite = MetricSuite(include_frequency_bands=config.include_frequency_bands)
    metrics = suite(prediction, target)
    if config.task == "rollout":
        metrics.update(
            rollout_horizon_metrics(prediction, target, horizons=config.horizons, time_axis=1)
        )
        if config.include_stability:
            metrics.update(stability_metrics(prediction, reference=target, time_axis=1))
    if config.physical_representation:
        metrics.update(
            physical_errors(
                prediction,
                target,
                representation=config.physical_representation,
                channel_axis=config.channel_axis,
                spatial_axes=config.spatial_axes,
                valid_mask=None if geometry is None else np.asarray(geometry) < 0.5,
            )
        )
    return {key: float(value) for key, value in metrics.items()}


def run_inference(
    method: Any,
    loader: Iterable[Any],
    *,
    config: EvaluationConfig | Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run deterministic inference and optionally save compressed arrays."""

    if config is None:
        config = EvaluationConfig()
    elif isinstance(config, Mapping):
        config = EvaluationConfig(**dict(config))
    device = resolve_device(config.device) if _is_torch_model(method) else None
    predictions, targets = [], []
    for batch in loader:
        prediction, target = predict_batch(method, batch, config, device)
        predictions.append(prediction)
        targets.append(target)
    if not predictions:
        raise ValueError("Data loader produced no batches")
    all_predictions = np.concatenate(predictions, axis=0)
    all_targets = np.concatenate(targets, axis=0)
    if config.predictions_path:
        destination = Path(config.predictions_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, prediction=all_predictions, target=all_targets)
    return all_predictions, all_targets


class _HDF5PredictionWriter(AbstractContextManager["_HDF5PredictionWriter"]):
    """Append prediction batches to one atomically published HDF5 file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(
                "streamed predictions require an .h5 or .hdf5 output path; "
                "use run_inference() directly when an in-memory NPZ is desired"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.path.with_suffix(self.path.suffix + ".partial")
        self.handle = h5py.File(self.partial, "w")
        self.count = 0

    def _append_array(
        self,
        name: str,
        values: np.ndarray,
        start: int,
        stop: int,
    ) -> None:
        if len(values) != stop - start:
            raise ValueError(f"{name} batch length differs from prediction batch")
        if name not in self.handle:
            tail = tuple(int(size) for size in values.shape[1:])
            chunks = (min(16, len(values)), *tail)
            self.handle.create_dataset(
                name,
                shape=(0, *tail),
                maxshape=(None, *tail),
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
                dtype=values.dtype,
            )
        dataset = self.handle[name]
        if tuple(dataset.shape[1:]) != tuple(values.shape[1:]):
            raise ValueError(
                f"inconsistent {name} batch shape {values.shape[1:]}; expected {dataset.shape[1:]}"
            )
        dataset.resize(stop, axis=0)
        dataset[start:stop] = values

    def _append_strings(self, name: str, values: list[str], start: int, stop: int) -> None:
        if len(values) != stop - start:
            raise ValueError(f"{name} batch length differs from prediction batch")
        if name not in self.handle:
            self.handle.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                chunks=(min(64, len(values)),),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
        dataset = self.handle[name]
        dataset.resize(stop, axis=0)
        dataset[start:stop] = values

    def append(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        *,
        observation: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        geometry: np.ndarray | None = None,
        metadata: Iterable[Mapping[str, Any]] | None = None,
    ) -> None:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target batches must have identical shapes")
        if prediction.ndim < 1 or len(prediction) < 1:
            raise ValueError("prediction batches must contain at least one sample")
        start, stop = self.count, self.count + len(prediction)
        self._append_array("prediction", prediction, start, stop)
        self._append_array("target", target, start, stop)
        for name, values in (
            ("observation", observation),
            ("mask", mask),
            ("geometry", geometry),
        ):
            if values is not None:
                self._append_array(name, np.asarray(values), start, stop)
        if metadata is not None:
            rows = [dict(row) for row in metadata]
            self._append_strings(
                "metadata_json",
                [json.dumps(row, sort_keys=True, default=str) for row in rows],
                start,
                stop,
            )
            self._append_strings(
                "sample_id",
                [str(row.get("sample_id", "")) for row in rows],
                start,
                stop,
            )
        self.count = stop

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.handle.attrs["samples"] = self.count
        self.handle.flush()
        self.handle.close()
        if exc_type is None:
            os.replace(self.partial, self.path)
        return False


def _accumulate_sample_metrics(
    totals: dict[str, float],
    prediction: np.ndarray,
    target: np.ndarray,
    config: EvaluationConfig,
    geometry: np.ndarray | None = None,
) -> None:
    """Accumulate official per-sample scores independently of batch boundaries."""

    for index in range(len(prediction)):
        sample_metrics = evaluate_predictions(
            prediction[index : index + 1],
            target[index : index + 1],
            config,
            geometry=None if geometry is None else geometry[index : index + 1],
        )
        for key, value in sample_metrics.items():
            if key == "max_norm_growth":
                totals[key] = max(totals.get(key, float("-inf")), value)
            else:
                totals[key] = totals.get(key, 0.0) + value


def evaluate_model(
    method: Any,
    loader: Iterable[Any],
    *,
    config: EvaluationConfig | Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run inference, compute the official metric groups, and emit JSON."""

    if config is None:
        config = EvaluationConfig()
    elif isinstance(config, Mapping):
        config = EvaluationConfig(**dict(config))
    # Full-tier fields are never concatenated here. Metrics are defined as
    # means of per-sample scores, which makes them independent of DataLoader
    # batch size. Prediction exports use an append-only HDF5 writer.
    device = resolve_device(config.device) if _is_torch_model(method) else None
    totals: dict[str, float] = {}
    sample_count = 0
    writer: _HDF5PredictionWriter | None = None
    try:
        if config.predictions_path:
            writer = _HDF5PredictionWriter(config.predictions_path)
        for batch in loader:
            prediction, target = predict_batch(method, batch, config, device)
            raw_input, raw_mask, _, raw_geometry, raw_metadata = unpack_batch_context(batch)
            observation = (
                None if raw_input is None else _numpy_channels_last(raw_input, config.data_layout)
            )
            exported_mask = (
                None if raw_mask is None else _numpy_channels_last(raw_mask, config.data_layout)
            )
            geometry = (
                None
                if raw_geometry is None
                else _numpy_channels_last(raw_geometry, config.data_layout)
            )
            _accumulate_sample_metrics(totals, prediction, target, config, geometry)
            if writer is not None:
                writer.append(
                    prediction,
                    target,
                    observation=observation,
                    mask=exported_mask,
                    geometry=geometry,
                    metadata=raw_metadata,
                )
            sample_count += len(prediction)
    except BaseException as exc:
        if writer is not None:
            writer.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        if writer is not None:
            writer.__exit__(None, None, None)
    if not sample_count:
        raise ValueError("Data loader produced no batches")
    metrics = {
        key: value if key == "max_norm_growth" else value / sample_count
        for key, value in totals.items()
    }
    result: dict[str, Any] = {
        **dict(context or {}),
        "method": getattr(method, "name", method.__class__.__name__),
        "task": config.task,
        "samples": sample_count,
        "metrics": metrics,
        "evaluation_config": asdict(config),
    }
    if config.report_path:
        destination = Path(config.report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
    return result


def _requested_metric_view(
    available: Mapping[str, float], requested: Iterable[str] | None
) -> tuple[dict[str, float], dict[str, str]]:
    """Select friendly metric groups without inventing unavailable physics scores."""

    if requested is None:
        return dict(available), {}
    aliases = {
        "rel_l2": "relative_l2",
        "relative_l2_error": "relative_l2",
        "l2": "relative_l2",
    }
    selected: dict[str, float] = {}
    unavailable: dict[str, str] = {}
    for raw_name in requested:
        name = str(raw_name).strip().lower().replace("-", "_")
        if not name:
            continue
        if name == "pde_residual":
            unavailable[name] = (
                "a PDE residual needs the validated family-specific discrete operator, "
                "condition, geometry, and physical parameters; prediction/target arrays "
                "alone are insufficient"
            )
            continue
        if name == "spectral":
            matches = {
                key: value
                for key, value in available.items()
                if key.startswith("spectral_")
                or key in {"spectral_centroid_error", "high_frequency_energy_error"}
            }
            if not matches:
                unavailable[name] = "no spectral metrics were produced for this prediction file"
            selected.update(matches)
            continue
        canonical = aliases.get(name, name)
        if canonical not in available:
            unavailable[name] = f"metric {canonical!r} is not available for this task"
            continue
        selected[canonical] = float(available[canonical])
    if not selected and not unavailable:
        raise ValueError("--metrics did not contain any metric names")
    return selected, unavailable


def evaluate_prediction_file(
    predictions_path: str | Path,
    *,
    task: str = "recovery",
    metrics: Iterable[str] | None = None,
    report_path: str | Path | None = None,
    data_root: str | Path | None = None,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Stream an inference HDF5 file and write an honest standalone report.

    This is the implementation behind ``pdeobs eval --pred ...``.  Unsupported
    requested metrics are reported explicitly instead of being silently dropped
    or replaced with an invalid proxy.
    """

    source = Path(predictions_path).expanduser().resolve()
    if source.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("--pred must reference an HDF5 file produced by pdeobs infer")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    canonical_task = str(task).strip().lower().replace("-", "_")
    canonical_task = {
        "sparse_recovery": "recovery",
        "sparse_to_full_recovery": "recovery",
        "world_model": "rollout",
        "world_modeling": "rollout",
    }.get(canonical_task, canonical_task)
    requested_metrics = None if metrics is None else tuple(metrics)
    destination = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else source.with_name(f"{source.stem}.metrics.json")
    )
    records_destination = destination.with_name(f"{destination.stem}.records.jsonl")
    records_partial = records_destination.with_name(f"{records_destination.name}.partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    config = EvaluationConfig(task=canonical_task)
    totals: dict[str, float] = {}
    sample_count = 0
    with h5py.File(source, "r") as handle, records_partial.open("w", encoding="utf-8") as rows:
        missing = {name for name in ("prediction", "target") if name not in handle}
        if missing:
            raise ValueError(f"prediction file is missing datasets: {sorted(missing)}")
        prediction = handle["prediction"]
        target = handle["target"]
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target datasets differ: {prediction.shape} != {target.shape}"
            )
        if not len(prediction):
            raise ValueError("prediction file contains no samples")
        geometry = handle.get("geometry")
        metadata_json = handle.get("metadata_json")
        sample_ids = handle.get("sample_id")
        for start in range(0, len(prediction), int(batch_size)):
            stop = min(len(prediction), start + int(batch_size))
            predicted_batch = np.asarray(prediction[start:stop])
            target_batch = np.asarray(target[start:stop])
            geometry_batch = None if geometry is None else np.asarray(geometry[start:stop])
            metadata_rows = (
                None if metadata_json is None else metadata_json.asstr()[start:stop].tolist()
            )
            id_rows = None if sample_ids is None else sample_ids.asstr()[start:stop].tolist()
            for offset in range(stop - start):
                sample_metrics = evaluate_predictions(
                    predicted_batch[offset : offset + 1],
                    target_batch[offset : offset + 1],
                    config,
                    geometry=(
                        None if geometry_batch is None else geometry_batch[offset : offset + 1]
                    ),
                )
                for key, value in sample_metrics.items():
                    if key == "max_norm_growth":
                        totals[key] = max(totals.get(key, float("-inf")), value)
                    else:
                        totals[key] = totals.get(key, 0.0) + value
                record: dict[str, Any] = {}
                if metadata_rows is not None:
                    loaded_metadata = json.loads(metadata_rows[offset])
                    if isinstance(loaded_metadata, dict):
                        record.update(loaded_metadata)
                if id_rows is not None and id_rows[offset]:
                    record.setdefault("sample_id", id_rows[offset])
                record.update(
                    {
                        "row_index": start + offset,
                        "task": canonical_task,
                        "prediction_file": str(source),
                        "metrics": sample_metrics,
                    }
                )
                rows.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                sample_count += 1
    os.replace(records_partial, records_destination)
    averaged = {
        key: value if key == "max_norm_growth" else value / sample_count
        for key, value in totals.items()
    }
    selected, unavailable = _requested_metric_view(averaged, requested_metrics)
    result: dict[str, Any] = {
        "schema_version": 1,
        "task": canonical_task,
        "prediction_file": str(source),
        "data_root": None if data_root is None else str(Path(data_root)),
        "samples": sample_count,
        "metrics": selected,
        "requested_metrics": None if requested_metrics is None else list(requested_metrics),
        "unavailable_metrics": unavailable,
        "sample_records": str(records_destination),
        "output": str(destination),
    }
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def ood_metric_degradation(
    iid_metrics: Mapping[str, float],
    ood_metrics: Mapping[str, float],
    *,
    higher_is_better: Mapping[str, bool] | None = None,
    mode: str = "ratio",
) -> dict[str, float]:
    orientation = dict(higher_is_better or {})
    shared = sorted(set(iid_metrics) & set(ood_metrics))
    return {
        f"{name}_ood_degradation": ood_degradation(
            iid_metrics[name],
            ood_metrics[name],
            higher_is_better=orientation.get(name, False),
            mode=mode,
        )
        for name in shared
    }


def evaluate_ood(
    method: Any,
    iid_loader: Iterable[Any],
    ood_loader: Iterable[Any],
    *,
    config: EvaluationConfig | Mapping[str, Any] | None = None,
    higher_is_better: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Evaluate paired IID/OOD loaders and report oriented degradation."""

    iid = evaluate_model(method, iid_loader, config=config, context={"split": "iid"})
    ood = evaluate_model(method, ood_loader, config=config, context={"split": "ood"})
    return {
        "iid": iid,
        "ood": ood,
        "degradation": ood_metric_degradation(
            iid["metrics"], ood["metrics"], higher_is_better=higher_is_better
        ),
    }


inference = run_inference
evaluate = evaluate_model
