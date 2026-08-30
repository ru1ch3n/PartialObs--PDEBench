# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Dependency-free CSV/JSON result aggregation for cluster sweeps."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def flatten_record(
    record: Mapping[str, Any], *, separator: str = ".", prefix: str = ""
) -> dict[str, Any]:
    """Flatten nested config/metric mappings while leaving lists JSON-encoded."""

    output: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{separator}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            output.update(flatten_record(value, separator=separator, prefix=name))
        elif isinstance(value, (list, tuple, dict)):
            output[name] = json.dumps(value, sort_keys=True)
        else:
            output[name] = value
    return output


def _records_from_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("records"), list):
        records = payload["records"]
    elif isinstance(payload, Mapping):
        records = [payload]
    else:
        raise ValueError(f"Unsupported JSON report structure in {path}")
    return [flatten_record(record) for record in records]


def _coerce(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            number = float(value)
            return number if math.isfinite(number) else value
        except ValueError:
            return value


def load_records(paths: str | Path | Iterable[str | Path]) -> list[dict[str, Any]]:
    """Read any mixture of JSON and CSV run artifacts."""

    if isinstance(paths, (str, Path)):
        paths = [paths]
    records: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        if path.suffix.lower() == ".json":
            loaded = _records_from_json(path)
        elif path.suffix.lower() == ".jsonl":
            loaded = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f"JSONL record {line_number} in {path} must be an object")
                    loaded.append(flatten_record(record))
        elif path.suffix.lower() == ".csv":
            with path.open("r", newline="", encoding="utf-8") as handle:
                loaded = [
                    {key: _coerce(value) for key, value in row.items()}
                    for row in csv.DictReader(handle)
                ]
        else:
            raise ValueError(f"Report must be .json, .jsonl, or .csv: {path}")
        for record in loaded:
            record.setdefault("source_file", str(path))
        records.extend(loaded)
    return records


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    group_by: Sequence[str] = ("method", "task", "split"),
    metrics: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute count/mean/std/min/max for every numeric metric per group."""

    flattened = [flatten_record(record) for record in records]
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in flattened:
        groups[tuple(record.get(key) for key in group_by)].append(record)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        selected_metrics = (
            list(metrics)
            if metrics is not None
            else sorted(
                {
                    name
                    for row in rows
                    for name, value in row.items()
                    if name not in group_by and _is_number(value)
                }
            )
        )
        aggregate = {name: value for name, value in zip(group_by, key, strict=True)}
        aggregate["runs"] = len(rows)
        for metric in selected_metrics:
            values = [float(row[metric]) for row in rows if _is_number(row.get(metric))]
            if not values:
                continue
            aggregate[f"{metric}.count"] = len(values)
            aggregate[f"{metric}.mean"] = mean(values)
            aggregate[f"{metric}.std"] = pstdev(values) if len(values) > 1 else 0.0
            aggregate[f"{metric}.min"] = min(values)
            aggregate[f"{metric}.max"] = max(values)
        output.append(aggregate)
    return output


def write_json_report(
    records: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: Any = (
        {"metadata": dict(metadata or {}), "records": list(records)}
        if metadata is not None
        else list(records)
    )
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return destination


def write_csv_report(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for record in records for key in record})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in record.items()
                }
            )
    return destination


def aggregate_reports(
    inputs: str | Path | Iterable[str | Path],
    *,
    group_by: Sequence[str] = ("method", "task", "split"),
    metrics: Sequence[str] | None = None,
    json_path: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load, aggregate, and optionally emit both publication-friendly formats."""

    aggregated = aggregate_records(load_records(inputs), group_by=group_by, metrics=metrics)
    if json_path is not None:
        write_json_report(aggregated, json_path)
    if csv_path is not None:
        write_csv_report(aggregated, csv_path)
    return aggregated


summarize_reports = aggregate_reports
