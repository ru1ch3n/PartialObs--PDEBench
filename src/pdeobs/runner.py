"""Config adapters used by the high-level CLI commands."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    apply_overrides,
    config_hash,
    expand_environment,
    load_config,
    save_resolved_config,
)
from .dataset import BenchmarkDataset, collate_benchmark, find_shards
from .provenance import write_provenance

_FACTOR_FIELDS = ("pde", "boundary", "setting", "regime")


def _load_runner_config(
    source: str | Path | Mapping[str, Any], overrides: Sequence[str] = ()
) -> dict[str, Any]:
    """Load YAML or an inline, wheel-safe preset mapping with identical semantics."""

    if isinstance(source, Mapping):
        return expand_environment(apply_overrides(dict(source), overrides))
    return load_config(source, overrides)


def _positive_horizons(value: Any, *, name: str) -> tuple[int, ...]:
    """Normalize one horizon scalar/sequence and reject ambiguous empty values."""

    values = (value,) if isinstance(value, (int, np.integer)) else tuple(value)
    horizons = tuple(sorted({int(item) for item in values}))
    if not horizons or horizons[0] < 1:
        raise ValueError(f"{name} must contain positive integers")
    return horizons


def _evaluation_horizons(config: Mapping[str, Any]) -> tuple[int, ...]:
    """Return horizons reported at evaluation, independently of training."""

    data = config.get("data", {})
    evaluation = config.get("evaluation", {})
    value = evaluation.get(
        "horizons",
        data.get(
            "evaluation_horizons", data.get("rollout_horizons", data.get("horizon", (1, 2, 4, 8)))
        ),
    )
    return _positive_horizons(value, name="evaluation horizons")


def _training_horizons(config: Mapping[str, Any]) -> tuple[int, ...]:
    """Return rollout horizons visible to optimization.

    Legacy ``data.rollout_horizons`` is now treated as an evaluation schedule.
    When a rollout config does not explicitly name its training schedule, the
    official short-horizon protocol uses the available horizons up to two.
    """

    data = config.get("data", {})
    value = data.get("training_horizons", data.get("training_horizon"))
    if value is None:
        evaluation = _evaluation_horizons(config)
        short = tuple(horizon for horizon in evaluation if horizon <= 2)
        return short or (evaluation[0],)
    return _positive_horizons(value, name="training horizons")


def _run_directory(config: Mapping[str, Any], output: str | Path | None) -> Path:
    if output is not None:
        path = Path(output)
    else:
        root = Path(config.get("output", {}).get("root", "runs"))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = root / f"{config.get('name', 'run')}-{timestamp}-{config_hash(config)[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _method(config: Mapping[str, Any]) -> Any:
    from .methods import create_method, create_model

    method_config = dict(config.get("method", {}))
    name = str(method_config.pop("name", "unet"))
    kwargs = dict(method_config.pop("kwargs", {}) or {})
    base = method_config.pop("base", None)
    if method_config:
        raise ValueError(f"Unknown method configuration keys: {sorted(method_config)}")
    if base is not None:
        base = dict(base)
        base_name = str(base.pop("name"))
        base_kwargs = dict(base.pop("kwargs", {}) or {})
        if base:
            raise ValueError(f"Unknown base-method keys: {sorted(base)}")
        kwargs["one_step_model"] = create_model(base_name, **base_kwargs)
    try:
        return create_model(name, **kwargs)
    except KeyError:
        return create_method(name, **kwargs)


def _data_settings(config: Mapping[str, Any]) -> tuple[list[Path], dict[str, Any]]:
    data = dict(config.get("data", {}))
    root = data.pop("root", "datasets")
    pattern = data.pop("train_glob", data.pop("glob", "**/*.h5"))
    shards = find_shards(root, pattern)
    if not shards:
        raise FileNotFoundError(f"No HDF5 shards match {Path(root) / pattern}")
    return shards, data


def _allow_split_fallback(config: Mapping[str, Any]) -> bool:
    """Return the explicit small-data escape hatch for missing official splits."""

    value = config.get("data", {}).get("allow_split_fallback", False)
    if not isinstance(value, bool):
        raise ValueError("data.allow_split_fallback must be true or false")
    return value


def _dataset(
    config: Mapping[str, Any],
    subset: str | None,
    *,
    fallback_unfiltered: bool = False,
    ood_view: str | None = None,
    ood_membership: bool | None = None,
    mask_override: Mapping[str, Any] | None = None,
    rollout_horizon: int | None = None,
) -> BenchmarkDataset | None:
    shards, data = _data_settings(config)
    task = str(config.get("task", "recovery"))
    filters = dict(data.get("filters", {}))
    configured_view = str(ood_view or data.get("ood_view", data.get("split", "iid")))
    configured_view = {
        "boundary_ood": "boundary",
        "setting_ood": "setting",
        "parameter_ood": "parameter",
        "combination_ood": "combination",
    }.get(configured_view, configured_view)
    official_views = {"boundary", "setting", "parameter", "combination"}
    if configured_view == "iid":
        split = subset
    elif configured_view in official_views:
        split = subset
        membership = subset == "test" if ood_membership is None else bool(ood_membership)
        filters[f"{configured_view}_ood"] = membership
    elif configured_view in {"train", "validation", "test"}:
        split = configured_view
    else:
        raise ValueError(
            "data.ood_view/data.split must be iid, boundary, setting, parameter, "
            "combination, train, validation, or test"
        )
    mask = dict(mask_override or data.get("mask", {}))
    target_step = int(data.get("target_step", data.get("target_time", -1)))
    if task == "rollout":
        if rollout_horizon is not None:
            horizon = _positive_horizons(rollout_horizon, name="rollout horizon override")[-1]
        elif subset in {"train", "validation"}:
            horizon = _training_horizons(config)[-1]
        else:
            horizon = _evaluation_horizons(config)[-1]
    else:
        horizon = int(data.get("horizon", 8))
    history_steps = int(data.get("input_horizon", data.get("history_steps", 1)))
    options = dict(
        task=task,
        split=split,
        filters=filters,
        mask=mask,
        target_step=target_step,
        horizon=horizon,
        history_steps=history_steps,
        state_representation=str(data.get("state_representation", "native")),
        seed=int(config.get("seed", 0)),
        max_samples=data.get("max_samples"),
        verify=bool(data.get("verify_shards", False)),
    )
    try:
        dataset = BenchmarkDataset(shards, **options)
        dataset.effective_split = split
        return dataset
    except ValueError as exc:
        if fallback_unfiltered and split is not None:
            warnings.warn(
                f"No {split} samples; using all matching samples for this small run: {exc}",
                stacklevel=2,
            )
            options["split"] = None
            dataset = BenchmarkDataset(shards, **options)
            dataset.effective_split = None
            return dataset
        if subset == "validation":
            warnings.warn(
                f"Validation subset unavailable; training without it: {exc}",
                stacklevel=2,
            )
            return None
        raise


def _loader(dataset: BenchmarkDataset, config: Mapping[str, Any], *, shuffle: bool) -> Any:
    training = config.get("training", {})
    batch_size = min(int(training.get("batch_size", 16)), len(dataset))
    try:
        from torch.utils.data import DataLoader
    except ImportError:
        # Transparent NumPy baselines remain usable with the core installation.
        class NumpyBatchLoader:
            def __iter__(self):
                indices = np.arange(len(dataset))
                if shuffle:
                    np.random.default_rng(int(config.get("seed", 0))).shuffle(indices)
                for start in range(0, len(indices), batch_size):
                    yield collate_benchmark(
                        [dataset[int(i)] for i in indices[start : start + batch_size]]
                    )

        return NumpyBatchLoader()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=bool(training.get("pin_memory", True)),
        persistent_workers=bool(training.get("persistent_workers", False))
        and int(training.get("num_workers", 0)) > 0,
        collate_fn=collate_benchmark,
    )


def _dataset_context(dataset: BenchmarkDataset) -> dict[str, Any]:
    """Summarize immutable factor values represented by an evaluated dataset."""

    context: dict[str, Any] = {}
    for field in _FACTOR_FIELDS:
        values = sorted(
            {
                str(metadata[field])
                for metadata in dataset.metadata
                if metadata.get(field) is not None
            }
        )
        if len(values) == 1:
            context[field] = values[0]
        elif values:
            context[f"{field}_values"] = values
    return context


def _evaluation_context(
    config: Mapping[str, Any],
    dataset: BenchmarkDataset,
    **values: Any,
) -> dict[str, Any]:
    context = {"config_hash": config_hash(config), **_dataset_context(dataset)}
    context.update(values)
    return context


def _mask_context(mask: Mapping[str, Any], dataset: BenchmarkDataset) -> dict[str, Any]:
    protocol = str(mask.get("protocol", "random_3pct"))
    context: dict[str, Any] = {"mask_protocol": protocol}
    if mask.get("ratio") is not None:
        context["observation_ratio"] = float(mask["ratio"])
    elif protocol in {
        "random_1pct",
        "random_3pct",
        "random_5pct",
        "random_10pct",
    }:
        context["observation_ratio"] = {
            "random_1pct": 0.01,
            "random_3pct": 0.03,
            "random_5pct": 0.05,
            "random_10pct": 0.10,
        }[protocol]
    elif mask.get("count") is not None and dataset.metadata:
        resolution = dataset.metadata[0].get("resolution")
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            total = int(resolution[0]) * int(resolution[1])
            if total > 0:
                context["observation_ratio"] = float(mask["count"]) / total
    return context


def _training_config(config: Mapping[str, Any], run_dir: Path, resume: Path | None) -> Any:
    from .training import TrainingConfig

    values = dict(config.get("training", {}))
    for key in ("batch_size", "num_workers", "pin_memory", "persistent_workers"):
        values.pop(key, None)
    if "mixed_precision" in values:
        values["amp"] = values.pop("mixed_precision")
    values.setdefault("task", config.get("task", "recovery"))
    values.setdefault("seed", config.get("seed", 0))
    data = config.get("data", {})
    values.setdefault("target_step", data.get("target_step", data.get("target_time", -1)))
    if str(config.get("task", "recovery")) == "rollout":
        values.setdefault("horizon", _training_horizons(config)[-1])
    values.setdefault("history_steps", data.get("input_horizon", data.get("history_steps", 1)))
    if str(config.get("task", "recovery")) == "rollout":
        values.setdefault("rollout_target_offset", 0)
    values["checkpoint_dir"] = str(run_dir / "checkpoints")
    if resume is not None:
        values["resume_from"] = str(resume)
    known = {item.name for item in fields(TrainingConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown training keys: {sorted(unknown)}")
    return TrainingConfig(**values)


def _evaluation_config(
    config: Mapping[str, Any],
    *,
    predictions_path: Path | None = None,
    report_path: Path | None = None,
) -> Any:
    from .evaluation import EvaluationConfig

    values = dict(config.get("evaluation", {}))
    values.setdefault("task", config.get("task", "recovery"))
    data = config.get("data", {})
    values.setdefault("target_step", data.get("target_step", data.get("target_time", -1)))
    if str(config.get("task", "recovery")) == "rollout":
        horizons = _evaluation_horizons(config)
        values.setdefault("horizons", horizons)
        values.setdefault("horizon", horizons[-1])
    values.setdefault("history_steps", data.get("input_horizon", data.get("history_steps", 1)))
    if str(config.get("task", "recovery")) == "rollout":
        values.setdefault("rollout_target_offset", 0)
    if predictions_path is not None:
        values["predictions_path"] = str(predictions_path)
    if report_path is not None:
        values["report_path"] = str(report_path)
    known = {item.name for item in fields(EvaluationConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown evaluation keys: {sorted(unknown)}")
    return EvaluationConfig(**values)


def _load_checkpoint(model: Any, checkpoint: str | Path | None, device: str = "cpu") -> None:
    try:
        from torch import nn
    except ImportError as exc:
        if checkpoint is None:
            return
        raise ImportError("Loading a learned checkpoint requires PyTorch") from exc
    if checkpoint is None:
        if isinstance(model, nn.Module):
            raise ValueError(
                "inference/evaluation of a learned method requires --checkpoint/--ckpt; "
                "refusing to score randomly initialized weights"
            )
        return
    if not isinstance(model, nn.Module):
        raise TypeError("A checkpoint can only be loaded into a PyTorch method")
    from .training import load_checkpoint_payload

    payload = load_checkpoint_payload(checkpoint, map_location=device)
    state = payload.get("model_state", payload)
    model.load_state_dict(state)


def _prepare(
    config_path: str | Path | Mapping[str, Any],
    overrides: Sequence[str],
    output: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    config = _load_runner_config(config_path, overrides)
    run_dir = _run_directory(config, output)
    save_resolved_config(config, run_dir / "resolved.yaml")
    write_provenance(run_dir / "provenance.json", config=config)
    return config, run_dir


def run_train(
    *,
    config_path: str | Path | Mapping[str, Any],
    overrides: Sequence[str] = (),
    output: str | Path | None = None,
    checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    del checkpoint
    config, run_dir = _prepare(config_path, overrides, output)
    if dry_run:
        return {"run_dir": str(run_dir), "config_hash": config_hash(config), "dry_run": True}
    from .training import Trainer

    train_data = _dataset(
        config,
        "train",
        fallback_unfiltered=_allow_split_fallback(config),
    )
    assert train_data is not None
    validation_data = _dataset(config, "validation")
    model = _method(config)
    trainer = Trainer(model, _training_config(config, run_dir, Path(resume) if resume else None))
    history = trainer.fit(
        _loader(train_data, config, shuffle=True),
        _loader(validation_data, config, shuffle=False) if validation_data is not None else None,
    )
    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "checkpoints" / "best.pt"),
        "epochs_completed": len(history),
        "best_metric": trainer.best_metric,
        "samples": len(train_data),
    }


def run_infer(
    *,
    config_path: str | Path | Mapping[str, Any],
    overrides: Sequence[str] = (),
    output: str | Path | None = None,
    checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    del resume
    if output is None:
        raise ValueError("inference requires --output")
    config = _load_runner_config(config_path, overrides)
    output_path = Path(output).resolve()
    save_resolved_config(config, output_path.parent / "inference.resolved.yaml")
    write_provenance(output_path.parent / "inference.provenance.json", config=config)
    if dry_run:
        return {"output": str(output_path), "dry_run": True}
    from .evaluation import evaluate_model

    dataset = _dataset(
        config,
        "test",
        fallback_unfiltered=_allow_split_fallback(config),
    )
    assert dataset is not None
    model = _method(config)
    _load_checkpoint(model, checkpoint)
    evaluation = _evaluation_config(config, predictions_path=output_path)
    result = evaluate_model(
        model,
        _loader(dataset, config, shuffle=False),
        config=evaluation,
        context=_evaluation_context(
            config,
            dataset,
            split="test" if dataset.effective_split == "test" else "all_fallback",
        ),
    )
    return {
        "output": str(output_path),
        "samples": int(result["samples"]),
        "metrics": result["metrics"],
    }


def run_evaluate(
    *,
    config_path: str | Path | Mapping[str, Any],
    overrides: Sequence[str] = (),
    output: str | Path | None = None,
    checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    del resume
    config = _load_runner_config(config_path, overrides)
    report = Path(output).resolve() if output else _run_directory(config, None) / "metrics.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, report.parent / "evaluation.resolved.yaml")
    write_provenance(report.parent / "evaluation.provenance.json", config=config)
    if dry_run:
        return {"output": str(report), "dry_run": True}
    from .evaluation import evaluate_model, ood_metric_degradation
    from .splits import time_horizon_ood_split

    model = _method(config)
    _load_checkpoint(model, checkpoint)
    evaluation = _evaluation_config(config, report_path=report)
    aliases = {
        "boundary_ood": "boundary",
        "setting_ood": "setting",
        "parameter_ood": "parameter",
        "combination_ood": "combination",
        "horizon": "time_horizon",
        "horizon_ood": "time_horizon",
        "time_horizon_ood": "time_horizon",
    }
    views = tuple(
        dict.fromkeys(
            aliases.get(
                str(view).lower().replace("-", "_"),
                str(view).lower().replace("-", "_"),
            )
            for view in evaluation.ood_views
        )
    )
    valid_views = {"boundary", "setting", "parameter", "combination", "time_horizon"}
    invalid_views = set(views) - valid_views
    if invalid_views:
        raise ValueError(f"Unknown evaluation OOD views: {sorted(invalid_views)}")
    if not views:
        configured_view = str(config.get("data", {}).get("ood_view", "iid"))
        if configured_view in valid_views - {"time_horizon"}:
            views = (configured_view,)
    configured_factor_view = str(config.get("data", {}).get("ood_view", "iid"))
    factor_iid_view = (
        configured_factor_view
        if configured_factor_view in valid_views - {"time_horizon"}
        else "iid"
    )
    factor_iid_context = (
        {"factor_ood_view": factor_iid_view, "factor_is_ood": False}
        if factor_iid_view != "iid"
        else {}
    )
    mask_protocols = tuple(str(protocol) for protocol in evaluation.mask_protocols)
    if not views and not mask_protocols:
        dataset = _dataset(
            config,
            "test",
            fallback_unfiltered=_allow_split_fallback(config),
        )
        assert dataset is not None
        training_mask = dict(config.get("data", {}).get("mask", {"protocol": "random_3pct"}))
        return evaluate_model(
            model,
            _loader(dataset, config, shuffle=False),
            config=evaluation,
            context=_evaluation_context(
                config,
                dataset,
                split="test" if dataset.effective_split == "test" else "all_fallback",
                **_mask_context(training_mask, dataset),
            ),
        )

    child_evaluation = replace(
        evaluation,
        report_path=None,
        predictions_path=None,
        ood_views=(),
        mask_protocols=(),
    )
    result: dict[str, Any] = {
        "config_hash": config_hash(config),
        "method": getattr(model, "name", model.__class__.__name__),
        "task": child_evaluation.task,
        "ood": {},
        "mask_ood": {},
    }
    training_mask = dict(config.get("data", {}).get("mask", {"protocol": "random_3pct"}))
    for view in (view for view in views if view != "time_horizon"):
        iid_data = _dataset(config, "test", ood_view=view, ood_membership=False)
        ood_data = _dataset(config, "test", ood_view=view, ood_membership=True)
        assert iid_data is not None and ood_data is not None
        iid = evaluate_model(
            model,
            _loader(iid_data, config, shuffle=False),
            config=child_evaluation,
            context=_evaluation_context(
                config,
                iid_data,
                split="iid",
                ood_view=view,
                is_ood=False,
                **_mask_context(training_mask, iid_data),
            ),
        )
        ood = evaluate_model(
            model,
            _loader(ood_data, config, shuffle=False),
            config=child_evaluation,
            context=_evaluation_context(
                config,
                ood_data,
                split="ood",
                ood_view=view,
                is_ood=True,
                **_mask_context(training_mask, ood_data),
            ),
        )
        result["ood"][view] = {
            "iid": iid,
            "ood": ood,
            "degradation": ood_metric_degradation(iid["metrics"], ood["metrics"]),
        }

    if "time_horizon" in views:
        training_horizons = _training_horizons(config)
        evaluation_horizons = _evaluation_horizons(config)
        requested_horizons = tuple(sorted(set(training_horizons) | set(evaluation_horizons)))
        horizon_data = _dataset(
            config,
            "test",
            fallback_unfiltered=_allow_split_fallback(config),
            ood_view=factor_iid_view,
            ood_membership=False if factor_iid_view != "iid" else None,
            rollout_horizon=requested_horizons[-1],
        )
        assert horizon_data is not None
        horizon_results: dict[int, dict[str, Any]] = {}
        for horizon in requested_horizons:
            horizon_evaluation = replace(
                child_evaluation,
                horizon=horizon,
                horizons=(horizon,),
            )
            membership = time_horizon_ood_split(horizon, training_horizons=training_horizons)
            horizon_result = evaluate_model(
                model,
                _loader(horizon_data, config, shuffle=False),
                config=horizon_evaluation,
                context=_evaluation_context(
                    config,
                    horizon_data,
                    split="iid" if membership == "train" else "ood",
                    ood_view="time_horizon",
                    is_ood=membership != "train",
                    rollout_horizon=horizon,
                    **factor_iid_context,
                    **_mask_context(training_mask, horizon_data),
                ),
            )
            for metric in ("rel_l2", "mse"):
                endpoint = f"{metric}_h{horizon}"
                if endpoint not in horizon_result["metrics"]:
                    raise ValueError(
                        f"trajectory is too short to evaluate requested horizon {horizon}"
                    )
                horizon_result["metrics"][f"{metric}_at_horizon"] = horizon_result["metrics"][
                    endpoint
                ]
            horizon_results[horizon] = horizon_result

        reference_horizon = training_horizons[-1]
        reference = horizon_results[reference_horizon]
        horizon_summary: dict[str, Any] = {
            "training_horizons": list(training_horizons),
            "evaluation_horizons": list(evaluation_horizons),
            "reference_horizon": reference_horizon,
            "iid": {},
            "ood": {},
        }
        for horizon, horizon_result in horizon_results.items():
            if horizon in training_horizons:
                horizon_summary["iid"][str(horizon)] = horizon_result
            else:
                horizon_summary["ood"][str(horizon)] = {
                    "result": horizon_result,
                    "degradation": ood_metric_degradation(
                        reference["metrics"], horizon_result["metrics"]
                    ),
                }
        result["ood"]["time_horizon"] = horizon_summary

    if mask_protocols:
        iid_data = _dataset(
            config,
            "test",
            ood_view=factor_iid_view,
            ood_membership=False if factor_iid_view != "iid" else None,
            mask_override=training_mask,
        )
        assert iid_data is not None
        iid = evaluate_model(
            model,
            _loader(iid_data, config, shuffle=False),
            config=child_evaluation,
            context=_evaluation_context(
                config,
                iid_data,
                split="iid",
                ood_view="mask",
                is_ood=False,
                **factor_iid_context,
                **_mask_context(training_mask, iid_data),
            ),
        )
        result["mask_ood"]["iid"] = iid
        for protocol in mask_protocols:
            mask = {"protocol": protocol}
            ood_data = _dataset(
                config,
                "test",
                ood_view=factor_iid_view,
                ood_membership=False if factor_iid_view != "iid" else None,
                mask_override=mask,
            )
            assert ood_data is not None
            ood = evaluate_model(
                model,
                _loader(ood_data, config, shuffle=False),
                config=child_evaluation,
                context=_evaluation_context(
                    config,
                    ood_data,
                    split="ood",
                    ood_view="mask",
                    is_ood=True,
                    **factor_iid_context,
                    **_mask_context(mask, ood_data),
                ),
            )
            result["mask_ood"][protocol] = {
                "result": ood,
                "degradation": ood_metric_degradation(iid["metrics"], ood["metrics"]),
            }

    report.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _configured_factor_context(config: Mapping[str, Any]) -> dict[str, Any]:
    filters = config.get("data", {}).get("filters", {})
    context: dict[str, Any] = {}
    if isinstance(filters, Mapping):
        for field in _FACTOR_FIELDS:
            value = filters.get(field)
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                context[field] = value
    return context


def _metric_record(
    result: Mapping[str, Any],
    *,
    base: Mapping[str, Any],
    **dimensions: Any,
) -> dict[str, Any] | None:
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    record = dict(base)
    for key, value in result.items():
        if key in {"metrics", "evaluation_config"} or isinstance(value, Mapping):
            continue
        record[key] = value
    record.update(dimensions)
    record["metrics"] = dict(metrics)
    return record


def _evaluation_metric_records(
    evaluation: Mapping[str, Any],
    *,
    experiment_index: int,
    experiment_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Flatten every IID/OOD result into analyzer- and leaderboard-ready rows."""

    base = {
        "experiment_index": experiment_index,
        "experiment": str(experiment_config.get("name", f"run-{experiment_index:03d}")),
        **_configured_factor_context(experiment_config),
    }
    records: list[dict[str, Any]] = []
    direct = _metric_record(evaluation, base=base)
    if direct is not None:
        records.append(direct)

    ood_results = evaluation.get("ood", {})
    if isinstance(ood_results, Mapping):
        for view in sorted(ood_results):
            payload = ood_results[view]
            if not isinstance(payload, Mapping):
                continue
            if view == "time_horizon":
                iid = payload.get("iid", {})
                if isinstance(iid, Mapping):
                    for horizon, value in sorted(iid.items(), key=lambda item: int(item[0])):
                        if isinstance(value, Mapping):
                            row = _metric_record(
                                value,
                                base=base,
                                split="iid",
                                ood_view=view,
                                is_ood=False,
                                rollout_horizon=int(horizon),
                            )
                            if row is not None:
                                records.append(row)
                ood = payload.get("ood", {})
                if isinstance(ood, Mapping):
                    for horizon, value in sorted(ood.items(), key=lambda item: int(item[0])):
                        nested = value.get("result") if isinstance(value, Mapping) else None
                        if isinstance(nested, Mapping):
                            row = _metric_record(
                                nested,
                                base=base,
                                split="ood",
                                ood_view=view,
                                is_ood=True,
                                rollout_horizon=int(horizon),
                            )
                            if row is not None:
                                records.append(row)
                continue
            for split, is_ood in (("iid", False), ("ood", True)):
                value = payload.get(split)
                if isinstance(value, Mapping):
                    row = _metric_record(
                        value,
                        base=base,
                        split=split,
                        ood_view=view,
                        is_ood=is_ood,
                    )
                    if row is not None:
                        records.append(row)

    mask_results = evaluation.get("mask_ood", {})
    if isinstance(mask_results, Mapping):
        iid = mask_results.get("iid")
        if isinstance(iid, Mapping):
            row = _metric_record(
                iid,
                base=base,
                split="iid",
                ood_view="mask",
                is_ood=False,
            )
            if row is not None:
                records.append(row)
        for protocol in sorted(key for key in mask_results if key != "iid"):
            value = mask_results[protocol]
            nested = value.get("result") if isinstance(value, Mapping) else None
            if isinstance(nested, Mapping):
                row = _metric_record(
                    nested,
                    base=base,
                    split="ood",
                    ood_view="mask",
                    is_ood=True,
                    mask_protocol=protocol,
                )
                if row is not None:
                    records.append(row)
    return records


def run_benchmark(
    *,
    config_path: str | Path | Mapping[str, Any],
    overrides: Sequence[str] = (),
    output: str | Path | None = None,
    checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    del checkpoint, resume
    config = _load_runner_config(config_path, overrides)
    experiments = config.get("experiments")
    if not experiments:
        return run_train(
            config_path=config_path,
            overrides=overrides,
            output=output,
            dry_run=dry_run,
        )
    root = _run_directory(config, output)
    results = []
    metric_records = []
    for index, experiment in enumerate(experiments):
        if isinstance(experiment, str):
            experiment = {"config": experiment}
        experiment_source = experiment["config"]
        if isinstance(experiment_source, Mapping):
            experiment_path: str | Path | Mapping[str, Any] = experiment_source
            experiment_label = str(experiment.get("name", f"inline-{index:03d}"))
        else:
            resolved_path = Path(str(experiment_source))
            if not resolved_path.is_absolute():
                base = (
                    Path.cwd()
                    if isinstance(config_path, Mapping)
                    else Path(config_path).resolve().parent
                )
                resolved_path = base / resolved_path
            experiment_path = resolved_path
            experiment_label = str(resolved_path)
        mode = str(experiment.get("mode", "train_eval")).lower()
        if mode not in {"train", "eval", "train_eval"}:
            raise ValueError("benchmark experiment mode must be train, eval, or train_eval")
        run_output = root / f"run-{index:03d}"
        experiment_overrides = list(experiment.get("set", []))
        experiment_config = _load_runner_config(experiment_path, experiment_overrides)
        trained = None
        evaluation = None
        if mode in {"train", "train_eval"}:
            trained = run_train(
                config_path=experiment_path,
                overrides=experiment_overrides,
                output=run_output,
                dry_run=dry_run,
            )
        if mode in {"eval", "train_eval"} and not dry_run:
            selected_checkpoint = experiment.get("checkpoint")
            if mode == "train_eval":
                assert trained is not None
                selected_checkpoint = trained.get("checkpoint")
            evaluation = run_evaluate(
                config_path=experiment_path,
                overrides=experiment_overrides,
                output=run_output / "metrics.json",
                checkpoint=selected_checkpoint,
            )
            metric_records.extend(
                _evaluation_metric_records(
                    evaluation,
                    experiment_index=index,
                    experiment_config=experiment_config,
                )
            )
        result = {
            "index": index,
            "mode": mode,
            "config": experiment_label,
            "training": trained,
            "evaluation": evaluation,
        }
        results.append(result)
    from .reports import aggregate_records, flatten_record, write_csv_report, write_json_report

    analysis_json = root / "analysis_records.json"
    analysis_csv = root / "analysis_records.csv"
    write_json_report(metric_records, analysis_json)
    write_csv_report([flatten_record(record) for record in metric_records], analysis_csv)

    leaderboard = []
    if metric_records:
        metric_names = sorted(
            {
                key
                for record in metric_records
                for key in flatten_record(record)
                if key.startswith("metrics.")
            }
        )
        leaderboard = aggregate_records(
            metric_records,
            group_by=(
                "method",
                "task",
                "pde",
                "boundary",
                "setting",
                "regime",
                "ood_view",
                "split",
                "mask_protocol",
                "rollout_horizon",
            ),
            metrics=metric_names,
        )
        write_json_report(leaderboard, root / "leaderboard.json")
        write_csv_report(leaderboard, root / "leaderboard.csv")
    summary = {
        "benchmark_dir": str(root),
        "runs": results,
        "leaderboard": leaderboard,
        "analysis_records": {
            "count": len(metric_records),
            "json": str(analysis_json),
            "csv": str(analysis_csv),
        },
        "dry_run": dry_run,
    }
    (root / "benchmark.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary
