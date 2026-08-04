from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from pdeobs.aggregate import ShardValidationError, summarize_dataset, validate_hdf5_shard
from pdeobs.schema import Sample
from pdeobs.storage import AtomicHDF5ShardWriter, shard_sidecars, write_jsonl_manifest


def _write_valid_shard(path: Path, *, count: int = 2) -> Path:
    spec = {
        "pde": "poisson",
        "boundary": "periodic",
        "setting": "smooth_grf",
        "regime": "low",
        "sample_start": 0,
        "sample_count": count,
        "shard_index": 0,
        "output_path": str(path),
        "seed": 7,
        "provenance": {
            "captured_at_utc": "2026-01-01T00:00:00+00:00",
            "config_hash": "config-a",
            "git": {"commit": "commit-a", "dirty": False, "status": ""},
            "runtime": {"hostname": "node-a", "python": "3.12.1"},
        },
    }
    with AtomicHDF5ShardWriter(path, expected_count=count, spec=spec) as writer:
        for index in range(count):
            writer.append(
                Sample(
                    condition=np.full((4, 4), index, dtype=np.float32),
                    trajectory=np.full((1, 4, 4), index + 1, dtype=np.float32),
                    geometry=np.zeros((4, 4), dtype=np.float32),
                    metadata={
                        "sample_id": f"{path.stem}-sample-{index}",
                        "pde": "poisson",
                        "boundary": "periodic",
                        "setting": "smooth_grf",
                        "regime": "low",
                        "split": "train",
                        "seed": index,
                    },
                )
            )
    return path


def test_validate_and_summarize_small_shard(tmp_path: Path) -> None:
    path = _write_valid_shard(tmp_path / "fixture.h5")

    result = validate_hdf5_shard(path)
    summary = summarize_dataset(tmp_path, validate=True)

    assert result["samples"] == 2
    assert summary["shard_count"] == 1
    assert summary["sample_count"] == 2
    assert summary["samples_by_family"] == {"poisson": 2}


def test_strict_summary_rejects_zero_shards(tmp_path: Path) -> None:
    with pytest.raises(ShardValidationError, match="no HDF5 shards"):
        summarize_dataset(tmp_path, validate=True)


@pytest.mark.parametrize(
    ("component", "message"),
    (
        ("manifest", "missing completion manifest"),
        ("checksum", "missing checksum sidecar"),
        ("metadata_csv", "missing metadata_csv sidecar"),
        ("metadata_json", "missing metadata_json sidecar"),
        ("metadata_dataset", "lacks datasets"),
        ("schema", "HDF5 schema differs"),
    ),
)
def test_strict_validation_rejects_missing_integrity_components(
    tmp_path: Path, component: str, message: str
) -> None:
    path = _write_valid_shard(tmp_path / component / "fixture.h5")
    sidecars = shard_sidecars(path)
    if component in sidecars:
        sidecars[component].unlink()
    elif component == "metadata_dataset":
        with h5py.File(path, "a") as handle:
            del handle["metadata"]
    elif component == "schema":
        with h5py.File(path, "a") as handle:
            del handle.attrs["schema_version"]
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(component)

    with pytest.raises(ShardValidationError, match=message):
        validate_hdf5_shard(path)


def test_expected_plan_requires_exact_paths_and_counts(tmp_path: Path) -> None:
    root = tmp_path / "data"
    path = _write_valid_shard(
        root / "poisson" / "periodic" / "smooth_grf" / "low" / "shard_00000.h5"
    )
    row = {
        "pde": "poisson",
        "boundary": "periodic",
        "setting": "smooth_grf",
        "regime": "low",
        "sample_start": 0,
        "sample_count": 2,
        "shard_index": 0,
        "output_path": str(tmp_path / "rebased" / path.name),
        "seed": 7,
        "provenance": {
            "captured_at_utc": "2026-01-02T00:00:00+00:00",
            "config_hash": "config-a",
            "git": {"commit": "commit-a", "dirty": False, "status": ""},
            "runtime": {"hostname": "node-b", "python": "3.12.1"},
            "slurm": {"SLURM_JOB_ID": "202"},
        },
    }
    plan = write_jsonl_manifest([row], tmp_path / "plan.jsonl")
    summary = summarize_dataset(root, validate=True, expected_plan=plan)
    assert summary["sample_count"] == 2

    write_jsonl_manifest([{**row, "seed": 8}], plan)
    with pytest.raises(ShardValidationError, match="plan spec differs"):
        summarize_dataset(root, validate=True, expected_plan=plan)

    changed_commit = {
        **row,
        "provenance": {
            **row["provenance"],
            "git": {"commit": "commit-b", "dirty": False, "status": ""},
        },
    }
    write_jsonl_manifest([changed_commit], plan)
    with pytest.raises(ShardValidationError, match="plan spec differs"):
        summarize_dataset(root, validate=True, expected_plan=plan)

    write_jsonl_manifest([{**row, "sample_count": 1}], plan)
    with pytest.raises(ShardValidationError, match="expected 1 rows"):
        summarize_dataset(root, validate=True, expected_plan=plan)

    missing = {
        **row,
        "shard_index": 1,
        "output_path": str(path.with_name("shard_00001.h5")),
    }
    write_jsonl_manifest([row, missing], plan)
    with pytest.raises(ShardValidationError, match="missing 1"):
        summarize_dataset(root, validate=True, expected_plan=plan)

    extra_path = _write_valid_shard(
        root / "poisson" / "periodic" / "smooth_grf" / "low" / "shard_00001.h5"
    )
    assert extra_path.is_file()
    write_jsonl_manifest([row], plan)
    with pytest.raises(ShardValidationError, match="unexpected 1"):
        summarize_dataset(root, validate=True, expected_plan=plan)
