# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Deterministic, dependency-free problem-difficulty summaries.

The analyzer consumes the flat or nested JSON/CSV records emitted by benchmark
runs.  It deliberately produces tables rather than figures so that paper plots
can be regenerated from versioned, machine-readable artifacts.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import yaml

from .reports import load_records, write_csv_report

SCHEMA_VERSION = "pdeobs.problem-difficulty/v1"

_DIMENSION_ALIASES: dict[str, tuple[str, ...]] = {
    "method": ("method", "method_name", "model", "model_name"),
    "task": ("task", "task_name"),
    "observation_ratio": (
        "observation_ratio",
        "observed_ratio",
        "observation_fraction",
        "observed_fraction",
        "obs_ratio",
        "mask_ratio",
        "data.mask.ratio",
        "mask.ratio",
    ),
    "observation_pattern": (
        "observation_pattern",
        "observation_protocol",
        "mask_pattern",
        "mask_protocol",
        "data.mask.protocol",
        "mask.protocol",
    ),
    "pde": ("pde", "pde_family", "family", "metadata.pde", "metadata.family"),
    "boundary": (
        "boundary",
        "boundary_type",
        "boundary_condition",
        "bc",
        "metadata.boundary",
    ),
    "setting": (
        "setting",
        "setting_name",
        "condition_setting",
        "metadata.setting",
    ),
    "regime": (
        "regime",
        "parameter_regime",
        "difficulty_regime",
        "metadata.regime",
    ),
    "rollout_horizon": (
        "rollout_horizon",
        "prediction_horizon",
        "forecast_horizon",
        "horizon",
    ),
    "split": ("split", "subset", "data_split"),
    "ood_view": ("ood_view", "ood_type", "shift", "shift_type"),
    "is_ood": ("is_ood", "ood", "ood_membership"),
    "sample_id": ("sample_id", "semantic_id", "id"),
    "run_id": ("run_id", "experiment_id"),
}

_SCALE_ALIASES: dict[str, tuple[str, ...]] = {
    "training_samples": (
        "training_samples",
        "train_samples",
        "train_size",
        "n_train",
        "dataset_size",
        "data.train_size",
    ),
    "parameter_count": (
        "parameter_count",
        "parameters_count",
        "num_parameters",
        "n_parameters",
        "model_parameters",
    ),
    "runtime_seconds": (
        "runtime_seconds",
        "elapsed_seconds",
        "wall_time_seconds",
        "train_seconds",
        "inference_seconds",
    ),
    "peak_memory_mb": (
        "peak_memory_mb",
        "max_memory_mb",
        "gpu_memory_mb",
        "memory_mb",
    ),
}

_DIFFICULTY_DIMENSIONS = (
    "observation_ratio",
    "observation_pattern",
    "pde",
    "boundary",
    "setting",
    "regime",
)

_METRIC_TOKENS = (
    "accuracy",
    "band",
    "correlation",
    "divergence",
    "energy",
    "error",
    "frequency",
    "likelihood",
    "loss",
    "mae",
    "mass",
    "metric",
    "mse",
    "nll",
    "psnr",
    "regret",
    "residual",
    "rmse",
    "rollout",
    "score",
    "spectral",
    "spectrum",
    "ssim",
    "stability",
    "vorticity",
    "_l1",
    "_l2",
    "l1_",
    "l2_",
    "rel_l2",
)

_SPECTRAL_TOKENS = ("spectral", "spectrum", "frequency", "fourier", "_band")
_HIGHER_IS_BETTER_TOKENS = (
    "accuracy",
    "correlation",
    "likelihood",
    "psnr",
    "r2",
    "score",
    "ssim",
)
_LOWER_IS_BETTER_TOKENS = ("error", "loss", "regret")
_RETRIEVAL_SCORE_METRIC = re.compile(r"(?:^|_)(?:recall|map|ndcg)(?:(?:@|_at_)(?:\d+|k))?(?:$|_)")
_HORIZON_METRIC = re.compile(r"^(?P<metric>.+?)(?:_horizon_|_h)(?P<horizon>\d+)$")


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for bounded problem-difficulty analysis."""

    primary_metric: str | None = None
    metrics: tuple[str, ...] = ()
    higher_is_better: tuple[str, ...] = ()
    group_anchor: tuple[str, ...] = ("method", "task")
    top_k: int = 20
    min_group_size: int = 1

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least one")
        if self.min_group_size < 1:
            raise ValueError("min_group_size must be at least one")


def load_analysis_config(path: str | Path | None) -> AnalysisConfig:
    """Load the optional YAML configuration used by :func:`analyze_path`."""

    if path is None:
        return AnalysisConfig()
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Analysis config must contain a mapping: {source}")
    values = payload.get("analysis", payload)
    if not isinstance(values, Mapping):
        raise ValueError("analysis config section must contain a mapping")
    metrics = values.get("metrics", ())
    if isinstance(metrics, str):
        metrics = (metrics,)
    higher = values.get("higher_is_better", ())
    if isinstance(higher, str):
        higher = (higher,)
    anchors = values.get("group_anchor", ("method", "task"))
    if isinstance(anchors, str):
        anchors = tuple(part.strip() for part in anchors.split(",") if part.strip())
    return AnalysisConfig(
        primary_metric=values.get("primary_metric"),
        metrics=tuple(str(item) for item in metrics),
        higher_is_better=tuple(str(item) for item in higher),
        group_anchor=tuple(str(item) for item in anchors),
        top_k=int(values.get("top_k", 20)),
        min_group_size=int(values.get("min_group_size", 1)),
    )


def _key_index(record: Mapping[str, Any]) -> dict[str, str]:
    return {str(key).lower(): str(key) for key in record}


def _lookup(record: Mapping[str, Any], aliases: Sequence[str]) -> tuple[str | None, Any | None]:
    index = _key_index(record)
    for alias in aliases:
        if alias.lower() in index:
            key = index[alias.lower()]
            return key, record[key]
    for alias in aliases:
        suffix = f".{alias.lower()}"
        matches = sorted(
            (key for lowered, key in index.items() if lowered.endswith(suffix)),
            key=lambda key: (key.count("."), key),
        )
        if matches:
            return matches[0], record[matches[0]]
    return None, None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, str):
        text = value.strip()
        percent = text.endswith("%")
        if percent:
            text = text[:-1]
        try:
            result = float(text)
        except ValueError:
            return None
        if not math.isfinite(result):
            return None
        return result / 100.0 if percent else result
    return None


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "ood"}:
            return True
        if lowered in {"false", "no", "0", "iid"}:
            return False
    return None


def _canonical_metric_name(name: str) -> str:
    value = name.strip().lower().replace("-", "_").replace(" ", "_")
    for prefix in ("evaluation.metrics.", "result.metrics.", "metrics."):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    return value.replace(".", "_")


def _looks_like_metric(raw_name: str, canonical_name: str) -> bool:
    lowered = raw_name.lower()
    if ".metrics." in lowered or lowered.startswith("metrics."):
        return True
    return any(token in canonical_name for token in _METRIC_TOKENS)


def _normalize_record(
    record: Mapping[str, Any], row_index: int, config: AnalysisConfig
) -> dict[str, Any]:
    dimensions: dict[str, Any] = {}
    consumed: set[str] = set()
    for name, aliases in _DIMENSION_ALIASES.items():
        key, value = _lookup(record, aliases)
        if key is None or value is None or value == "":
            continue
        consumed.add(key)
        if name == "observation_ratio":
            parsed = _number(value)
            if parsed is not None:
                value = parsed
        elif name == "rollout_horizon":
            parsed = _number(value)
            if parsed is not None:
                value = int(parsed)
        elif name == "is_ood":
            parsed_bool = _boolean(value)
            if parsed_bool is not None:
                value = parsed_bool
        dimensions[name] = value

    if "observation_ratio" not in dimensions:
        _, observed = _lookup(
            record, ("observed_points", "observation_count", "n_observed", "mask.count")
        )
        _, total = _lookup(record, ("grid_points", "spatial_points", "n_grid"))
        observed_number, total_number = _number(observed), _number(total)
        if observed_number is not None and total_number and total_number > 0:
            dimensions["observation_ratio"] = observed_number / total_number

    scales: dict[str, float] = {}
    for name, aliases in _SCALE_ALIASES.items():
        key, value = _lookup(record, aliases)
        parsed = _number(value)
        if key is not None and parsed is not None:
            consumed.add(key)
            scales[name] = parsed

    requested = {_canonical_metric_name(name) for name in config.metrics}
    metrics: dict[str, float] = {}
    for key in sorted(record):
        value = _number(record[key])
        if value is None or key in consumed:
            continue
        canonical = _canonical_metric_name(str(key))
        if requested:
            selected = canonical in requested or any(
                canonical.endswith(f"_{name}") for name in requested
            )
        else:
            selected = _looks_like_metric(str(key), canonical)
        if selected:
            metrics.setdefault(canonical, value)

    metric_key, metric_name = _lookup(record, ("metric", "metric_name"))
    value_key, metric_value = _lookup(record, ("value", "metric_value"))
    parsed_value = _number(metric_value)
    if metric_key is not None and value_key is not None and parsed_value is not None:
        canonical = _canonical_metric_name(str(metric_name))
        if not requested or canonical in requested:
            metrics[canonical] = parsed_value

    return {
        "dimensions": dimensions,
        "metrics": metrics,
        "raw": dict(record),
        "row_index": row_index,
        "scales": scales,
    }


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    return row["dimensions"].get(key, row["scales"].get(key))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _statistics(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.5),
        "q75": _quantile(values, 0.75),
        "max": max(values),
    }


def _present_keys(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[str]:
    return [key for key in keys if any(_row_value(row, key) is not None for row in rows)]


def _sort_key(values: Sequence[Any]) -> tuple[tuple[int, float | str], ...]:
    sortable: list[tuple[int, float | str]] = []
    for value in values:
        if value is None:
            sortable.append((0, ""))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            sortable.append((1, float(value)))
        else:
            sortable.append((2, str(value)))
    return tuple(sortable)


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    group_by: Sequence[str],
    metrics: Sequence[str],
    *,
    min_group_size: int,
    required: str | None = None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if required is not None and _row_value(row, required) is None:
            continue
        groups[tuple(_row_value(row, name) for name in group_by)].append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: _sort_key(item[0])):
        if len(members) < min_group_size:
            continue
        summary = {name: value for name, value in zip(group_by, key, strict=True)}
        summary["records"] = len(members)
        for metric in metrics:
            values = [row["metrics"][metric] for row in members if metric in row["metrics"]]
            if not values:
                continue
            for statistic, value in _statistics(values).items():
                summary[f"{metric}.{statistic}"] = value
        if any(name.endswith(".count") for name in summary):
            output.append(summary)
    return output


def _metric_direction(metric: str, config: AnalysisConfig) -> str:
    canonical_higher = {_canonical_metric_name(name) for name in config.higher_is_better}
    if metric in canonical_higher:
        return "higher_is_better"
    if any(metric.endswith(f"_{name}") for name in canonical_higher):
        return "higher_is_better"
    if any(token in metric for token in _LOWER_IS_BETTER_TOKENS):
        return "lower_is_better"
    if any(token in metric for token in _HIGHER_IS_BETTER_TOKENS) or (
        _RETRIEVAL_SCORE_METRIC.search(metric) is not None
    ):
        return "higher_is_better"
    return "lower_is_better"


def _choose_primary(metrics: Sequence[str], config: AnalysisConfig) -> str | None:
    if not metrics:
        return None
    if config.primary_metric:
        requested = _canonical_metric_name(config.primary_metric)
        if requested not in metrics:
            raise ValueError(
                f"primary metric {config.primary_metric!r} was not found; "
                f"detected: {', '.join(metrics)}"
            )
        return requested
    for preferred in ("rel_l2", "relative_l2", "rmse", "mse", "mae"):
        if preferred in metrics:
            return preferred
    errors = [name for name in metrics if "error" in name or "loss" in name]
    return errors[0] if errors else metrics[0]


def _rollout_summary(
    rows: Sequence[Mapping[str, Any]], anchors: Sequence[str], config: AnalysisConfig
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        horizon_metrics: dict[int, dict[str, float]] = defaultdict(dict)
        explicit_horizon = row["dimensions"].get("rollout_horizon")
        for metric, value in row["metrics"].items():
            match = _HORIZON_METRIC.match(metric)
            if match:
                horizon_metrics[int(match.group("horizon"))][match.group("metric")] = value
            elif explicit_horizon is not None:
                horizon_metrics[int(explicit_horizon)][metric] = value
        for horizon, metrics in horizon_metrics.items():
            dimensions = dict(row["dimensions"])
            dimensions["rollout_horizon"] = horizon
            expanded.append(
                {
                    "dimensions": dimensions,
                    "metrics": metrics,
                    "raw": row["raw"],
                    "row_index": row["row_index"],
                    "scales": row["scales"],
                }
            )
    metrics = sorted({metric for row in expanded for metric in row["metrics"]})
    group_by = list(anchors)
    if any(_row_value(row, "pde") is not None for row in expanded):
        group_by.append("pde")
    group_by.append("rollout_horizon")
    return _summarize(
        expanded,
        group_by,
        metrics,
        min_group_size=config.min_group_size,
        required="rollout_horizon",
    )


def _ood_memberships(row: Mapping[str, Any]) -> list[tuple[str, bool]]:
    dimensions, raw = row["dimensions"], row["raw"]
    memberships: set[tuple[str, bool]] = set()
    view = str(dimensions.get("ood_view", "overall"))
    explicit = _boolean(dimensions.get("is_ood"))
    if explicit is not None:
        memberships.add((view, explicit))
    split = str(dimensions.get("split", "")).lower()
    if split == "iid":
        memberships.add((view, False))
    elif split == "ood":
        memberships.add((view, True))
    elif split.endswith("_ood"):
        inferred_view = split.removesuffix("_ood")
        memberships.add((inferred_view, True))
    for key, value in raw.items():
        leaf = str(key).lower().split(".")[-1]
        if not leaf.endswith("_ood") or "degradation" in leaf:
            continue
        membership = _boolean(value)
        if membership is not None:
            memberships.add((leaf.removesuffix("_ood"), membership))
    return sorted(memberships)


def _ood_summary(
    rows: Sequence[Mapping[str, Any]], anchors: Sequence[str], config: AnalysisConfig
) -> list[dict[str, Any]]:
    group_keys = list(anchors)
    if any(_row_value(row, "pde") is not None for row in rows):
        group_keys.append("pde")
    paired: dict[tuple[Any, ...], dict[str, dict[bool, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        for view, membership in _ood_memberships(row):
            key = tuple(_row_value(row, name) for name in group_keys) + (view,)
            for metric, value in row["metrics"].items():
                paired[key][metric][membership].append(value)
    output: list[dict[str, Any]] = []
    for key, metric_buckets in sorted(paired.items(), key=lambda item: _sort_key(item[0])):
        for metric in sorted(metric_buckets):
            iid = metric_buckets[metric].get(False, [])
            ood = metric_buckets[metric].get(True, [])
            if not iid or not ood:
                continue
            iid_mean, ood_mean = mean(iid), mean(ood)
            direction = _metric_direction(metric, config)
            degradation = ood_mean - iid_mean
            if direction == "higher_is_better":
                degradation = -degradation
            row = {name: value for name, value in zip(group_keys, key[:-1], strict=True)}
            row.update(
                {
                    "kind": "paired",
                    "ood_view": key[-1],
                    "metric": metric,
                    "direction": direction,
                    "iid_count": len(iid),
                    "ood_count": len(ood),
                    "iid_mean": iid_mean,
                    "ood_mean": ood_mean,
                    "absolute_degradation": degradation,
                    "relative_degradation": degradation / max(abs(iid_mean), 1e-12),
                    "ood_to_iid_ratio": ood_mean / max(abs(iid_mean), 1e-12),
                }
            )
            output.append(row)

    degradation_metrics = sorted(
        {metric for row in rows for metric in row["metrics"] if "ood_degradation" in metric}
    )
    if degradation_metrics:
        precomputed = _summarize(
            rows,
            group_keys,
            degradation_metrics,
            min_group_size=config.min_group_size,
        )
        for row in precomputed:
            row["kind"] = "precomputed"
        output.extend(precomputed)
    return output


def _linear_slope(points: Sequence[tuple[float, float]]) -> float | None:
    valid = [(math.log(x), math.log(y)) for x, y in points if x > 0 and y > 0]
    if len(valid) < 2:
        return None
    x_mean = mean(point[0] for point in valid)
    y_mean = mean(point[1] for point in valid)
    denominator = sum((x - x_mean) ** 2 for x, _ in valid)
    if denominator == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in valid) / denominator


def _scaling_summary(
    rows: Sequence[Mapping[str, Any]],
    anchors: Sequence[str],
    metrics: Sequence[str],
    config: AnalysisConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    levels: list[dict[str, Any]] = []
    for variable in _SCALE_ALIASES:
        groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            value = row["scales"].get(variable)
            if value is None:
                continue
            key = tuple(_row_value(row, name) for name in anchors) + (value,)
            groups[key].append(row)
        for key, members in sorted(groups.items(), key=lambda item: _sort_key(item[0])):
            if len(members) < config.min_group_size:
                continue
            summary = {name: value for name, value in zip(anchors, key[:-1], strict=True)}
            summary.update(
                {"scale_variable": variable, "scale_value": key[-1], "records": len(members)}
            )
            for metric in metrics:
                values = [row["metrics"][metric] for row in members if metric in row["metrics"]]
                if values:
                    for statistic, value in _statistics(values).items():
                        summary[f"{metric}.{statistic}"] = value
            if any(name.endswith(".count") for name in summary):
                levels.append(summary)

    trends: list[dict[str, Any]] = []
    trend_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in levels:
        key = tuple(row.get(name) for name in anchors) + (row["scale_variable"],)
        trend_groups[key].append(row)
    for key, members in sorted(trend_groups.items(), key=lambda item: _sort_key(item[0])):
        members = sorted(members, key=lambda row: float(row["scale_value"]))
        for metric in metrics:
            points = [
                (float(row["scale_value"]), float(row[f"{metric}.mean"]))
                for row in members
                if f"{metric}.mean" in row
            ]
            if len(points) < 2:
                continue
            trend = {name: value for name, value in zip(anchors, key[:-1], strict=True)}
            slope = _linear_slope(points)
            trend.update(
                {
                    "scale_variable": key[-1],
                    "metric": metric,
                    "direction": _metric_direction(metric, config),
                    "levels": len(points),
                    "smallest_scale": points[0][0],
                    "largest_scale": points[-1][0],
                    "smallest_scale_mean": points[0][1],
                    "largest_scale_mean": points[-1][1],
                    "relative_change": (points[-1][1] - points[0][1])
                    / max(abs(points[0][1]), 1e-12),
                    "log_log_slope": slope,
                }
            )
            trends.append(trend)
    return levels, trends


def _failure_rankings(
    rows: Sequence[Mapping[str, Any]],
    anchors: Sequence[str],
    primary_metric: str | None,
    config: AnalysisConfig,
) -> dict[str, list[dict[str, Any]]]:
    if primary_metric is None:
        return {"groups": [], "records": []}
    direction = _metric_direction(primary_metric, config)
    group_keys = _present_keys(
        rows,
        tuple(anchors) + _DIFFICULTY_DIMENSIONS + ("rollout_horizon",),
    )
    grouped = _summarize(
        rows,
        group_keys,
        (primary_metric,),
        min_group_size=config.min_group_size,
    )
    mean_key = f"{primary_metric}.mean"

    def group_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
        value = float(row[mean_key])
        score = value if direction == "higher_is_better" else -value
        return (score,) + _sort_key([row.get(name) for name in group_keys])

    ranked_groups = sorted(grouped, key=group_rank)[: config.top_k]
    for rank, row in enumerate(ranked_groups, start=1):
        row["rank"] = rank
        row["primary_metric"] = primary_metric
        row["direction"] = direction

    record_rows: list[dict[str, Any]] = []
    for row in rows:
        if primary_metric not in row["metrics"]:
            continue
        result = {
            "value": row["metrics"][primary_metric],
            "primary_metric": primary_metric,
            "direction": direction,
            "row_index": row["row_index"],
        }
        for name in (
            "method",
            "task",
            "sample_id",
            "run_id",
            *_DIFFICULTY_DIMENSIONS,
            "rollout_horizon",
        ):
            value = _row_value(row, name)
            if value is not None:
                result[name] = value
        if row["raw"].get("source_file") is not None:
            result["source_file"] = row["raw"]["source_file"]
        record_rows.append(result)

    reverse = direction == "lower_is_better"
    record_rows.sort(
        key=lambda row: (
            -float(row["value"]) if reverse else float(row["value"]),
            str(row.get("sample_id", "")),
            int(row["row_index"]),
        )
    )
    ranked_records = record_rows[: config.top_k]
    for rank, row in enumerate(ranked_records, start=1):
        row["rank"] = rank
    return {"groups": ranked_groups, "records": ranked_records}


def analyze_records(
    records: Sequence[Mapping[str, Any]], config: AnalysisConfig | None = None
) -> dict[str, Any]:
    """Create a complete machine-readable difficulty report from metric records."""

    config = config or AnalysisConfig()
    rows = [_normalize_record(record, index, config) for index, record in enumerate(records)]
    metrics = sorted({metric for row in rows for metric in row["metrics"]})
    if not metrics:
        raise ValueError("No numeric metric fields were detected in the input records")
    primary_metric = _choose_primary(metrics, config)
    anchors = _present_keys(rows, config.group_anchor)

    by_dimension: dict[str, list[dict[str, Any]]] = {}
    for dimension in _DIFFICULTY_DIMENSIONS:
        by_dimension[dimension] = _summarize(
            rows,
            [*anchors, dimension],
            metrics,
            min_group_size=config.min_group_size,
            required=dimension,
        )

    spectral_metrics = [
        metric for metric in metrics if any(token in metric for token in _SPECTRAL_TOKENS)
    ]
    spectral_groups = list(anchors)
    if any(_row_value(row, "pde") is not None for row in rows):
        spectral_groups.append("pde")
    spectral = (
        _summarize(
            rows,
            spectral_groups,
            spectral_metrics,
            min_group_size=config.min_group_size,
        )
        if spectral_metrics
        else []
    )

    levels, trends = _scaling_summary(rows, anchors, metrics, config)
    source_files = sorted(
        {
            str(row["raw"]["source_file"])
            for row in rows
            if row["raw"].get("source_file") is not None
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_count": len(rows),
        "source_files": source_files,
        "configuration": {
            "group_anchor": list(config.group_anchor),
            "higher_is_better": list(config.higher_is_better),
            "min_group_size": config.min_group_size,
            "top_k": config.top_k,
        },
        "detected": {
            "dimensions": sorted(
                {
                    name
                    for row in rows
                    for name in row["dimensions"]
                    if name not in {"sample_id", "run_id"}
                }
            ),
            "metrics": metrics,
            "metric_directions": {metric: _metric_direction(metric, config) for metric in metrics},
            "primary_metric": primary_metric,
            "scaling_variables": sorted({name for row in rows for name in row["scales"]}),
        },
        "summaries": {
            "overall": _summarize(rows, anchors, metrics, min_group_size=config.min_group_size),
            "by_dimension": by_dimension,
            "rollout_horizon": _rollout_summary(rows, anchors, config),
            "spectral": spectral,
            "ood_degradation": _ood_summary(rows, anchors, config),
            "scaling": {"levels": levels, "trends": trends},
        },
        "failure_rankings": _failure_rankings(rows, anchors, primary_metric, config),
    }


def analysis_output_paths(output: str | Path) -> dict[str, Path]:
    """Return the deterministic JSON and CSV artifact paths for ``output``."""

    destination = Path(output)
    if destination.suffix.lower() == ".json":
        prefix = destination.with_suffix("")
        return {
            "json": destination,
            "dimensions_csv": prefix.with_name(f"{prefix.name}.dimensions.csv"),
            "rollout_csv": prefix.with_name(f"{prefix.name}.rollout.csv"),
            "spectral_csv": prefix.with_name(f"{prefix.name}.spectral.csv"),
            "ood_csv": prefix.with_name(f"{prefix.name}.ood.csv"),
            "scaling_csv": prefix.with_name(f"{prefix.name}.scaling.csv"),
            "scaling_trends_csv": prefix.with_name(f"{prefix.name}.scaling-trends.csv"),
            "failures_csv": prefix.with_name(f"{prefix.name}.failures.csv"),
        }
    return {
        "json": destination / "difficulty.json",
        "dimensions_csv": destination / "by-dimension.csv",
        "rollout_csv": destination / "rollout-horizon.csv",
        "spectral_csv": destination / "spectral.csv",
        "ood_csv": destination / "ood-degradation.csv",
        "scaling_csv": destination / "scaling.csv",
        "scaling_trends_csv": destination / "scaling-trends.csv",
        "failures_csv": destination / "failure-rankings.csv",
    }


def write_analysis(report: Mapping[str, Any], output: str | Path) -> dict[str, Path]:
    """Write the canonical JSON report and non-empty flat CSV sidecars."""

    paths = analysis_output_paths(output)
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summaries = report["summaries"]
    dimension_rows = [
        {"analysis_dimension": dimension, **row}
        for dimension, rows in summaries["by_dimension"].items()
        for row in rows
    ]
    tables = {
        "dimensions_csv": dimension_rows,
        "rollout_csv": summaries["rollout_horizon"],
        "spectral_csv": summaries["spectral"],
        "ood_csv": summaries["ood_degradation"],
        "scaling_csv": summaries["scaling"]["levels"],
        "scaling_trends_csv": summaries["scaling"]["trends"],
        "failures_csv": report["failure_rankings"]["groups"],
    }
    written = {"json": paths["json"]}
    for name, rows in tables.items():
        if rows:
            write_csv_report(rows, paths[name])
            written[name] = paths[name]
    return written


def analyze_path(
    input_path: str | Path,
    output: str | Path,
    *,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """Load records from one JSON/CSV artifact, analyze them, and write outputs."""

    source = Path(input_path)
    if not source.is_file():
        raise ValueError(f"Analysis input must be one JSON, JSONL, or CSV file: {source}")
    report = analyze_records(load_records(source), config=config)
    write_analysis(report, output)
    return report
