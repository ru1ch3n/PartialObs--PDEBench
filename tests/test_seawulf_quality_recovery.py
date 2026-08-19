from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "hpc" / "seawulf" / "prepare_quality_recovery.py"
    spec = importlib.util.spec_from_file_location("prepare_quality_recovery", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan_row(tmp_path: Path, *, pde: str, shard: int) -> dict[str, object]:
    return {
        "pde": pde,
        "boundary": "periodic",
        "setting": "smooth_grf",
        "regime": "low",
        "sample_start": 0,
        "sample_count": 2,
        "shard_index": shard,
        "output_path": str(
            tmp_path
            / "planned-elsewhere"
            / pde
            / "periodic"
            / "smooth_grf"
            / "low"
            / f"shard_{shard:05d}.h5"
        ),
        "time_steps": 267 if pde != "poisson" else 1,
    }


def _failure(dataset_root: Path, row: dict[str, object], *, loss: float) -> dict[str, object]:
    output = dataset_root.joinpath(*Path(str(row["output_path"])).parts[-5:])
    path = output.with_suffix(".quality-failures.jsonl")
    return {
        "path": path,
        "record": {"sample_id": (f"seed-1/{row['pde']}/periodic/smooth_grf/low/000001")},
        "pde": row["pde"],
        "boundary": "periodic",
        "setting": "smooth_grf",
        "regime": "low",
        "loss": loss,
    }


def test_pilot_keeps_logical_sample_and_refines_worst_temporal_case(tmp_path):
    module = _module()
    row = _plan_row(tmp_path, pde="burgers", shard=1)
    failures = [
        _failure(tmp_path / "dataset", row, loss=0.06),
        _failure(tmp_path / "dataset", row, loss=0.08),
    ]

    pilot, summary = module.build_pilot([row], failures, factor=2)

    assert len(pilot) == 1
    assert pilot[0]["sample_start"] == 1
    assert pilot[0]["sample_count"] == 1
    assert pilot[0]["time_steps"] == 533
    assert pilot[0]["shard_index"] == 90000
    assert summary["cases"][0]["original_loss"] == 0.08


def test_recovery_uses_actual_dataset_root_and_refines_temporal_only(tmp_path):
    module = _module()
    dataset_root = tmp_path / "dataset"
    complete = _plan_row(tmp_path, pde="poisson", shard=0)
    temporal = _plan_row(tmp_path, pde="burgers", shard=1)
    complete_path = dataset_root.joinpath(*Path(str(complete["output_path"])).parts[-5:])
    complete_path.parent.mkdir(parents=True)
    complete_path.write_bytes(b"h5")
    complete_path.with_suffix(".manifest.json").write_text("{}\n", encoding="utf-8")

    recovery, combined, summary = module.build_recovery(
        [complete, temporal],
        [],
        dataset_root=dataset_root,
        factor=2,
        refine_all_temporal=True,
    )

    assert recovery == [{**temporal, "time_steps": 533}]
    assert combined[0] == complete
    assert combined[1]["time_steps"] == 533
    assert summary["already_complete_rows"] == 1
    assert summary["recovery_rows"] == 1
    assert summary["refined_recovery_rows"] == 1


def test_quarantine_moves_only_incomplete_shard_artifacts(tmp_path):
    module = _module()
    dataset_root = tmp_path / "dataset"
    row = _plan_row(tmp_path, pde="burgers", shard=1)
    output = dataset_root.joinpath(*Path(str(row["output_path"])).parts[-5:])
    output.parent.mkdir(parents=True)
    output.with_suffix(".h5.partial").write_bytes(b"partial")
    output.with_suffix(".quality-failures.jsonl").write_text("{}\n", encoding="utf-8")
    quarantine = tmp_path / "quarantine"

    summary = module.quarantine_incomplete(
        [row], dataset_root=dataset_root, quarantine_dir=quarantine
    )

    assert summary["quarantined_files"] == 2
    assert not output.with_suffix(".h5.partial").exists()
    manifest = json.loads((quarantine / "quarantine_manifest.json").read_text(encoding="utf-8"))
    assert manifest["moved_file_count"] == 2
    assert all(Path(item["target"]).is_file() for item in manifest["moved"])
