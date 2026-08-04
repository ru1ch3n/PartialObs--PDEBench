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

from .config import config_hash, load_config, save_resolved_config
from .dataset import BenchmarkDataset, collate_benchmark, find_shards
from .provenance import write_provenance


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


def _dataset(
    config: Mapping[str, Any],
    subset: str | None,
    *,
    fallback_unfiltered: bool = False,
    ood_view: str | None = None,
    ood_membership: bool | None = None,
    mask_override: Mapping[str, Any] | None = None,
) -> BenchmarkDataset | None:
    shards, data = _data_settings(config)
    task = str(config.get("task", "recovery"))
    filters = dict(data.get("filters", {}))
    configured_view = str(ood_view or data.get("ood_view", data.get("split", "iid")))
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
    horizons = data.get("rollout_horizons", [data.get("horizon", 8)])
    horizon = max(int(item) for item in horizons)
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
    horizons = data.get("rollout_horizons")
    if horizons:
        values.setdefault("horizon", max(int(item) for item in horizons))
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
    horizons = data.get("rollout_horizons")
    if horizons:
        values.setdefault("horizons", tuple(int(item) for item in horizons))
        values.setdefault("horizon", max(int(item) for item in horizons))
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
    if checkpoint is None:
        return
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("Loading a learned checkpoint requires PyTorch") from exc
    if not isinstance(model, nn.Module):
        raise TypeError("A checkpoint can only be loaded into a PyTorch method")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state", payload)
    model.load_state_dict(state)


def _prepare(
    config_path: str | Path,
    overrides: Sequence[str],
    output: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path, overrides)
    run_dir = _run_directory(config, output)
    save_resolved_config(config, run_dir / "resolved.yaml")
    write_provenance(run_dir / "provenance.json", config=config)
    return config, run_dir


def run_train(
    *,
    config_path: str | Path,
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

    train_data = _dataset(config, "train", fallback_unfiltered=True)
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
    config_path: str | Path,
    overrides: Sequence[str] = (),
    output: str | Path | None = None,
    checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    del resume
    if output is None:
        raise ValueError("inference requires --output")
    config = load_config(config_path, overrides)
    output_path = Path(output).resolve()
    save_resolved_config(config, output_path.parent / "inference.resolved.yaml")
    write_provenance(output_path.parent / "inference.provenance.json", config=config)
    if dry_run:
        return {"output": str(output_path), "dry_run": True}
    from .evaluation import evaluate_model

    dataset = _dataset(config, "test", fallback_unfiltered=True)
    assert dataset is not None
    model = _method(config)
    _load_checkpoint(model, checkpoint)
    evaluation = _evaluation_config(config, predictions_path=output_path)
    result = evaluate_model(
        model,
        _loader(dataset, config, shuffle=False),
        config=evaluation,
        context={
            "config_hash": config_hash(config),
            "split": "test" if dataset.effective_split == "test" else "all_fallback",
        },
    )
    return {
        "output": str(output_path),
        "samples": int(result["samples"]),
        "metrics": result["metrics"],
    }


def run_evaluate(
    *,
    config_path: str | Path,
    overrides: Sequence[str] = (),
    output: str | Path | None = None,
    checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    del resume
    config = load_config(config_path, overrides)
    report = Path(output).resolve() if output else _run_directory(config, None) / "metrics.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, report.parent / "evaluation.resolved.yaml")
    write_provenance(report.parent / "evaluation.provenance.json", config=config)
    if dry_run:
        return {"output": str(report), "dry_run": True}
    from .evaluation import evaluate_model, evaluate_ood, ood_metric_degradation

    model = _method(config)
    _load_checkpoint(model, checkpoint)
    evaluation = _evaluation_config(config, report_path=report)
    views = tuple(str(view) for view in evaluation.ood_views)
    mask_protocols = tuple(str(protocol) for protocol in evaluation.mask_protocols)
    if not views and not mask_protocols:
        dataset = _dataset(config, "test", fallback_unfiltered=True)
        assert dataset is not None
        return evaluate_model(
            model,
            _loader(dataset, config, shuffle=False),
            config=evaluation,
            context={
                "config_hash": config_hash(config),
                "split": "test" if dataset.effective_split == "test" else "all_fallback",
            },
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
    for view in views:
        iid_data = _dataset(config, "test", ood_view=view, ood_membership=False)
        ood_data = _dataset(config, "test", ood_view=view, ood_membership=True)
        assert iid_data is not None and ood_data is not None
        result["ood"][view] = evaluate_ood(
            model,
            _loader(iid_data, config, shuffle=False),
            _loader(ood_data, config, shuffle=False),
            config=child_evaluation,
        )

    if mask_protocols:
        training_mask = dict(config.get("data", {}).get("mask", {"protocol": "random_3pct"}))
        iid_data = _dataset(config, "test", mask_override=training_mask)
        assert iid_data is not None
        iid = evaluate_model(
            model,
            _loader(iid_data, config, shuffle=False),
            config=child_evaluation,
            context={"mask": training_mask.get("protocol", "random_3pct")},
        )
        result["mask_ood"]["iid"] = iid
        for protocol in mask_protocols:
            ood_data = _dataset(config, "test", mask_override={"protocol": protocol})
            assert ood_data is not None
            ood = evaluate_model(
                model,
                _loader(ood_data, config, shuffle=False),
                config=child_evaluation,
                context={"mask": protocol},
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


def run_benchmark(
    *,
    config_path: str | Path,
    overrides: Sequence[str] = (),
    output: str | Path | None = None,
    checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    del checkpoint, resume
    config = load_config(config_path, overrides)
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
        experiment_path = Path(str(experiment["config"]))
        if not experiment_path.is_absolute():
            experiment_path = Path(config_path).resolve().parent / experiment_path
        mode = str(experiment.get("mode", "train_eval")).lower()
        if mode not in {"train", "eval", "train_eval"}:
            raise ValueError("benchmark experiment mode must be train, eval, or train_eval")
        run_output = root / f"run-{index:03d}"
        experiment_overrides = list(experiment.get("set", []))
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
            if "metrics" in evaluation:
                metric_records.append(evaluation)
        result = {
            "index": index,
            "mode": mode,
            "config": str(experiment_path),
            "training": trained,
            "evaluation": evaluation,
        }
        results.append(result)
    leaderboard = []
    if metric_records:
        from .reports import aggregate_records, flatten_record, write_csv_report, write_json_report

        metric_names = sorted(
            {
                key
                for record in metric_records
                for key in flatten_record(record)
                if key.startswith("metrics.")
            }
        )
        leaderboard = aggregate_records(metric_records, metrics=metric_names)
        write_json_report(leaderboard, root / "leaderboard.json")
        write_csv_report(leaderboard, root / "leaderboard.csv")
    summary = {
        "benchmark_dir": str(root),
        "runs": results,
        "leaderboard": leaderboard,
        "dry_run": dry_run,
    }
    (root / "benchmark.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary
