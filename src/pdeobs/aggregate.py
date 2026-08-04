"""Dataset-shard validation and benchmark-result aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .reports import aggregate_records, load_records, write_csv_report
from .schema import SCHEMA_VERSION
from .storage import (
    StorageError,
    atomic_write_json,
    generation_specs_match,
    read_jsonl_manifest,
    shard_sidecars,
)


class ShardValidationError(RuntimeError):
    pass


_SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_completion(path: Path) -> dict[str, Any] | None:
    candidates = (
        path.with_suffix(".manifest.json"),
        path.with_suffix(path.suffix + ".complete.json"),
        path.with_suffix(".complete.json"),
        path.with_suffix(path.suffix + ".json"),
    )
    for candidate in candidates:
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and (
                "sha256" in payload or payload.get("status") == "complete"
            ):
                payload["_path"] = str(candidate)
                return payload
    return None


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _decoded_json(value: Any, *, source: str) -> Any:
    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(str(value))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShardValidationError(f"invalid JSON in {source}") from exc


def _required_integer(value: Any, *, source: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ShardValidationError(f"{source} must be an integer") from exc


def _strict_completion(
    shard: Path,
    *,
    count: int,
    attrs: Mapping[str, Any],
    metadata_rows: list[dict[str, Any]],
    verify_checksum: bool,
) -> tuple[dict[str, Any], str]:
    sidecars = shard_sidecars(shard)
    manifest_path = sidecars["manifest"]
    if not manifest_path.is_file():
        raise ShardValidationError(f"missing completion manifest for {shard}: {manifest_path}")
    try:
        completion = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShardValidationError(f"invalid completion manifest for {shard}") from exc
    if not isinstance(completion, dict):
        raise ShardValidationError(f"completion manifest for {shard} must be a mapping")
    if completion.get("status") != "complete":
        raise ShardValidationError(f"completion manifest for {shard} is not complete")
    if completion.get("schema_version") != SCHEMA_VERSION:
        raise ShardValidationError(f"completion manifest schema differs for {shard}")
    if completion.get("shard") != shard.name:
        raise ShardValidationError(f"completion manifest names the wrong shard for {shard}")
    if (
        _required_integer(
            completion.get("sample_count"), source=f"completion sample_count for {shard}"
        )
        != count
    ):
        raise ShardValidationError(f"completion manifest row count differs for {shard}")
    if (
        _required_integer(completion.get("bytes"), source=f"completion byte count for {shard}")
        != shard.stat().st_size
    ):
        raise ShardValidationError(f"completion manifest byte count differs for {shard}")
    manifest_digest = str(completion.get("sha256", ""))
    if not _SHA256_PATTERN.fullmatch(manifest_digest):
        raise ShardValidationError(f"completion manifest lacks a valid SHA-256 for {shard}")
    if not isinstance(completion.get("spec"), dict):
        raise ShardValidationError(f"completion manifest lacks a generation spec for {shard}")
    if not isinstance(attrs.get("spec"), dict) or _json_text(attrs["spec"]) != _json_text(
        completion["spec"]
    ):
        raise ShardValidationError(f"HDF5 and completion generation specs differ for {shard}")

    checksum_path = sidecars["checksum"]
    if not checksum_path.is_file():
        raise ShardValidationError(f"missing checksum sidecar for {shard}: {checksum_path}")
    try:
        checksum_parts = checksum_path.read_text(encoding="utf-8").strip().split()
    except OSError as exc:
        raise ShardValidationError(f"cannot read checksum sidecar for {shard}") from exc
    if not checksum_parts or not _SHA256_PATTERN.fullmatch(checksum_parts[0]):
        raise ShardValidationError(f"invalid checksum sidecar for {shard}")
    if checksum_parts[0].lower() != manifest_digest.lower():
        raise ShardValidationError(f"checksum sidecar and manifest differ for {shard}")
    if len(checksum_parts) > 1 and checksum_parts[-1].lstrip("*") != shard.name:
        raise ShardValidationError(f"checksum sidecar names the wrong shard for {shard}")

    for field, key in (("metadata_csv", "metadata_csv"), ("metadata_json", "metadata_json")):
        expected = sidecars[key]
        if completion.get(field) != expected.name or not expected.is_file():
            raise ShardValidationError(f"missing {field} sidecar for {shard}: {expected}")
    try:
        metadata_payload = json.loads(sidecars["metadata_json"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShardValidationError(f"invalid metadata JSON sidecar for {shard}") from exc
    if not isinstance(metadata_payload, dict):
        raise ShardValidationError(f"metadata JSON sidecar for {shard} must be a mapping")
    if (
        metadata_payload.get("schema_version") != SCHEMA_VERSION
        or metadata_payload.get("shard") != shard.name
        or not isinstance(metadata_payload.get("samples"), list)
        or _json_text(metadata_payload["samples"]) != _json_text(metadata_rows)
    ):
        raise ShardValidationError(f"metadata JSON sidecar does not match {shard}")
    try:
        with sidecars["metadata_csv"].open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            csv_rows = list(reader)
            csv_fields = tuple(reader.fieldnames or ())
    except OSError as exc:
        raise ShardValidationError(f"cannot read metadata CSV sidecar for {shard}") from exc
    flattened_rows = []
    for row in metadata_rows:
        flattened = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                value = _json_text(value)
            flattened[str(key)] = "" if value is None else str(value)
        flattened_rows.append(flattened)
    expected_fields = tuple(sorted({key for row in flattened_rows for key in row}))
    normalized_rows = [{key: row.get(key, "") for key in expected_fields} for row in flattened_rows]
    if csv_fields != expected_fields or csv_rows != normalized_rows:
        raise ShardValidationError(f"metadata CSV sidecar does not match {shard}")

    digest = _sha256(shard) if verify_checksum else manifest_digest
    if verify_checksum and digest.lower() != manifest_digest.lower():
        raise ShardValidationError(f"checksum mismatch for {shard}")
    completion = dict(completion)
    completion["_path"] = str(manifest_path)
    return completion, digest


def validate_hdf5_shard(
    path: str | Path,
    *,
    verify_checksum: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    shard = Path(path)
    arrays = ("condition", "trajectory", "geometry")
    required = (*arrays, "metadata") if strict else arrays
    metadata_rows: list[dict[str, Any]] = []
    try:
        with h5py.File(shard, "r") as handle:
            missing = [name for name in required if name not in handle]
            if missing:
                raise ShardValidationError(f"{shard} lacks datasets: {missing}")
            count = int(handle["condition"].shape[0])
            if count < 1:
                raise ShardValidationError(f"{shard} is empty")
            for name in required:
                dataset = handle[name]
                if int(dataset.shape[0]) != count:
                    raise ShardValidationError(f"{shard}:{name} has an inconsistent sample count")
            for name in arrays:
                dataset = handle[name]
                for start in range(0, count, 16):
                    stop = min(start + 16, count)
                    if not np.all(np.isfinite(dataset[start:stop])):
                        raise ShardValidationError(
                            f"{shard}:{name}[{start}:{stop}] contains a non-finite value"
                        )
            if handle["condition"].ndim != 4:
                raise ShardValidationError("condition must be NHWC")
            if handle["trajectory"].ndim != 5:
                raise ShardValidationError("trajectory must be NTHWC")
            if handle["geometry"].ndim != 4 or handle["geometry"].shape[-1] != 1:
                raise ShardValidationError("geometry must be NHW1")
            attrs = {
                str(key): value.item() if isinstance(value, np.generic) else value
                for key, value in handle.attrs.items()
            }
            if strict:
                if attrs.get("schema_version") != SCHEMA_VERSION:
                    raise ShardValidationError(f"HDF5 schema differs for {shard}")
                if (
                    _required_integer(
                        attrs.get("sample_count"), source=f"HDF5 sample_count for {shard}"
                    )
                    != count
                ):
                    raise ShardValidationError(f"HDF5 sample_count attribute differs for {shard}")
                if "spec_json" not in attrs:
                    raise ShardValidationError(f"HDF5 generation spec is missing for {shard}")
            if "spec_json" in attrs:
                attrs["spec"] = _decoded_json(attrs["spec_json"], source=f"{shard}:spec_json")
            if strict:
                if not isinstance(attrs.get("spec"), dict):
                    raise ShardValidationError(
                        f"HDF5 generation spec must be a mapping for {shard}"
                    )
                for index, encoded in enumerate(handle["metadata"]):
                    row = _decoded_json(encoded, source=f"{shard}:metadata[{index}]")
                    if not isinstance(row, dict):
                        raise ShardValidationError(
                            f"{shard}:metadata[{index}] must contain a JSON mapping"
                        )
                    metadata_rows.append(row)
                spec = attrs.get("spec", {})
                if isinstance(spec, Mapping) and spec.get("pde"):
                    required_metadata = {
                        "sample_id",
                        "pde",
                        "boundary",
                        "setting",
                        "regime",
                        "split",
                        "seed",
                    }
                    for index, row in enumerate(metadata_rows):
                        missing = sorted(required_metadata - row.keys())
                        if missing:
                            raise ShardValidationError(
                                f"{shard}:metadata[{index}] lacks official fields: {missing}"
                            )
            shapes = {name: list(handle[name].shape) for name in arrays}
    except ShardValidationError:
        raise
    except OSError as exc:
        raise ShardValidationError(f"Cannot read HDF5 shard {shard}: {exc}") from exc

    if strict:
        completion, digest = _strict_completion(
            shard,
            count=count,
            attrs=attrs,
            metadata_rows=metadata_rows,
            verify_checksum=verify_checksum,
        )
    else:
        completion = _read_completion(shard)
        digest = None
        if verify_checksum and completion and completion.get("sha256"):
            digest = _sha256(shard)
            if digest.lower() != str(completion["sha256"]).lower():
                raise ShardValidationError(f"Checksum mismatch for {shard}")
    return {
        "path": str(shard),
        "samples": count,
        "shapes": shapes,
        "attributes": attrs,
        "sha256": digest or (completion or {}).get("sha256"),
        "completion": completion,
        "_sample_ids": [
            str(row["sample_id"]) for row in metadata_rows if row.get("sample_id") is not None
        ],
    }


def _expected_plan_paths(
    root: Path, expected_plan: str | Path
) -> dict[Path, tuple[int, dict[str, Any]]]:
    try:
        rows = read_jsonl_manifest(expected_plan)
    except (OSError, StorageError) as exc:
        raise ShardValidationError(
            f"cannot read expected generation plan: {expected_plan}"
        ) from exc
    if not rows:
        raise ShardValidationError(f"expected generation plan is empty: {expected_plan}")
    root_resolved = root.resolve()
    expected: dict[Path, tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        family = row.get("pde", row.get("family"))
        required = {
            "family": family,
            "boundary": row.get("boundary"),
            "setting": row.get("setting"),
            "regime": row.get("regime"),
            "sample_count": row.get("sample_count"),
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ShardValidationError(
                f"expected plan row {index} lacks required fields: {', '.join(missing)}"
            )
        output_name = Path(str(row.get("output_path", ""))).name
        if not output_name:
            if row.get("shard_index") is None:
                raise ShardValidationError(
                    f"expected plan row {index} lacks output_path and shard_index"
                )
            shard_index = _required_integer(
                row["shard_index"], source=f"expected plan row {index} shard_index"
            )
            output_name = f"shard_{shard_index:05d}.h5"
        path = (
            root_resolved
            / str(family)
            / str(row["boundary"])
            / str(row["setting"])
            / str(row["regime"])
            / output_name
        ).resolve()
        if not path.is_relative_to(root_resolved):
            raise ShardValidationError(f"expected plan row {index} escapes dataset root")
        if path in expected:
            raise ShardValidationError(f"expected plan contains duplicate shard path: {path}")
        count = _required_integer(
            row["sample_count"], source=f"expected plan row {index} sample_count"
        )
        if count < 1:
            raise ShardValidationError(f"expected plan row {index} has an invalid sample_count")
        expected[path] = (count, dict(row))
    return expected


def summarize_dataset(
    root: str | Path,
    *,
    validate: bool = False,
    verify_checksum: bool = True,
    strict: bool = True,
    expected_plan: str | Path | None = None,
) -> dict[str, Any]:
    directory = Path(root)
    shards = sorted({*directory.rglob("*.h5"), *directory.rglob("*.hdf5")})
    if expected_plan is not None and not validate:
        raise ValueError("expected_plan requires validate=True")
    if validate and strict and not shards:
        raise ShardValidationError(f"no HDF5 shards found under {directory}")
    summaries = [
        validate_hdf5_shard(path, verify_checksum=verify_checksum, strict=strict)
        if validate
        else {"path": str(path), "samples": None}
        for path in shards
    ]
    if validate and strict:
        seen_ids: set[str] = set()
        for item in summaries:
            for sample_id in item.pop("_sample_ids", []):
                if sample_id in seen_ids:
                    raise ShardValidationError(f"duplicate sample_id across shards: {sample_id}")
                seen_ids.add(sample_id)
    if expected_plan is not None:
        expected = _expected_plan_paths(directory, expected_plan)
        actual = {Path(item["path"]).resolve(): item for item in summaries}
        missing = sorted(expected.keys() - actual.keys())
        unexpected = sorted(actual.keys() - expected.keys())
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing {len(missing)} (first: {missing[0]})")
            if unexpected:
                details.append(f"unexpected {len(unexpected)} (first: {unexpected[0]})")
            raise ShardValidationError("generation plan mismatch: " + "; ".join(details))
        for path, (expected_count, expected_spec) in expected.items():
            actual_count = int(actual[path]["samples"])
            if actual_count != expected_count:
                raise ShardValidationError(
                    f"generation plan expected {expected_count} rows in {path}, "
                    f"found {actual_count}"
                )
            actual_spec = actual[path].get("attributes", {}).get("spec")
            if not generation_specs_match(actual_spec, expected_spec):
                raise ShardValidationError(
                    f"generation plan spec differs from stored shard spec for {path}"
                )
    total = sum(item["samples"] or 0 for item in summaries)
    families = Counter()
    if validate:
        for item in summaries:
            attrs = item.get("attributes", {})
            spec = attrs.get("spec", {}) if isinstance(attrs.get("spec"), dict) else {}
            family = (
                attrs.get("pde") or attrs.get("family") or spec.get("pde") or spec.get("family")
            )
            if family:
                families[str(family)] += int(item["samples"])
    return {
        "root": str(directory.resolve()),
        "valid": True,
        "shard_count": len(shards),
        "sample_count": total if validate else None,
        "samples_by_family": dict(sorted(families.items())),
        "shards": summaries,
    }


def _report_files(root: Path) -> list[Path]:
    names = {"metrics.json", "results.json", "metrics.csv", "results.csv"}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.name in names)


def aggregate_path(
    input_root: str | Path,
    output: str | Path,
    *,
    validate_shards: bool = False,
    verify_checksum: bool = True,
    expected_plan: str | Path | None = None,
    group_by: Iterable[str] = ("method", "task", "split"),
) -> dict[str, Any]:
    root = Path(input_root)
    dataset = summarize_dataset(
        root,
        validate=validate_shards,
        verify_checksum=verify_checksum,
        expected_plan=expected_plan,
    )
    report_paths = _report_files(root)
    records = load_records(report_paths) if report_paths else []
    aggregated = aggregate_records(records, group_by=tuple(group_by)) if records else []
    payload = {
        "dataset": dataset,
        "report_files": [str(path) for path in report_paths],
        "leaderboard": aggregated,
    }
    destination = Path(output)
    atomic_write_json(destination, payload)
    if aggregated:
        write_csv_report(aggregated, destination.with_suffix(".csv"))
    return payload
