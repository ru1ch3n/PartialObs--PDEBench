# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pdeobs.cli import build_parser, main


def test_cli_lists_components(capsys) -> None:
    assert main(["list", "--kind", "pdes", "--json"]) == 0
    output = capsys.readouterr().out
    assert '"poisson"' in output
    assert '"navier_stokes"' in output


def test_download_one_line_uses_the_publication_gated_default_endpoint() -> None:
    args = build_parser().parse_args(["download", "--tier", "tiny", "--root", "data"])
    assert args.manifest is None
    assert args.root == Path("data")


def test_quality_profiles_and_gate_flags_parse() -> None:
    generated = build_parser().parse_args(
        [
            "generate",
            "--tier",
            "tiny",
            "--quality-profile",
            "strict",
            "--max-pde-loss",
            "0.25",
        ]
    )
    assert generated.quality_profile == "strict"
    assert generated.max_pde_loss == 0.25

    audited = build_parser().parse_args(
        [
            "quality",
            "--input",
            "data",
            "--output",
            "quality.json",
            "--strict",
            "--require-all-pdes",
            "--require-validated-solvers",
        ]
    )
    assert audited.quality_strict is True
    assert audited.require_all_pdes is True
    assert audited.require_validated_solvers is True


def test_cli_dispatches_doctor_download_and_aggregate(tmp_path: Path, monkeypatch, capsys) -> None:
    from pdeobs import aggregate, doctor, download
    from pdeobs.doctor import Check

    monkeypatch.setattr(
        doctor,
        "run_doctor",
        lambda **_: [Check("test runtime", True, "ready")],
    )
    assert main(["doctor"]) == 0
    assert "test runtime" in capsys.readouterr().out

    monkeypatch.setattr(
        download,
        "download_release",
        lambda manifest, output, tier, force=False: [Path(output) / f"{tier}.h5"],
    )
    assert (
        main(
            [
                "download",
                "--tier",
                "tiny",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "Verified 1 files" in capsys.readouterr().out

    monkeypatch.setattr(
        aggregate,
        "aggregate_path",
        lambda *_, **__: {"dataset": {"shard_count": 1}, "leaderboard": []},
    )
    assert (
        main(
            [
                "aggregate",
                "--input",
                str(tmp_path),
                "--output",
                str(tmp_path / "summary.json"),
            ]
        )
        == 0
    )
    assert "Found 1 shards" in capsys.readouterr().out


def test_cli_dispatches_quality_audit(tmp_path: Path, monkeypatch, capsys) -> None:
    from pdeobs import quality

    report = {
        "sample_count": 7,
        "shard_count": 2,
        "quality": {"by_pde": {}},
        "gate": {"status": "warning"},
    }
    monkeypatch.setattr(quality, "audit_dataset_quality", lambda *_, **__: report)
    output = tmp_path / "quality.json"

    assert main(["quality", "--input", str(tmp_path), "--output", str(output)]) == 0
    assert output.is_file()
    assert output.with_suffix(".csv").is_file()
    assert "Audited 7 samples" in capsys.readouterr().out


def test_cli_lists_builtins_in_fresh_process() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pdeobs", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    components = json.loads(completed.stdout)
    assert "poisson" in components["pdes"]
    assert "smooth_grf" in components["settings"]
    assert "random_3pct" in components["masks"]
    assert "relative_l2" in components["metrics"]
    assert "unet" in components["methods"]


def test_cli_plan_and_generation_dry_run(tmp_path: Path, capsys) -> None:
    config = tmp_path / "smoke.yaml"
    config.write_text(
        "\n".join(
            (
                "tier: tiny",
                "resolution: 8",
                "shard_size: 5",
                "seed: 3",
                "families: [poisson]",
                "boundaries: [periodic]",
                "settings: [smooth_grf]",
                "regimes: [low]",
                f"output: {{root: '{tmp_path.as_posix()}/data'}}",
            )
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "plan.jsonl"
    assert main(["plan", "--config", str(config), "--output", str(plan)]) == 0
    assert len(plan.read_text(encoding="utf-8").splitlines()) == 1
    assert (
        main(
            [
                "generate",
                "--config",
                str(config),
                "--output",
                str(tmp_path / "generated"),
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"selected_job_count": 1' in output


def test_config_free_generation_and_generate_case_contracts(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "generate",
                "--tier",
                "tiny",
                "--root",
                str(tmp_path),
                "--num-workers",
                "2",
                "--dry-run",
                "--set",
                "families=[poisson]",
                "--set",
                "boundaries=[periodic]",
                "--set",
                "settings=[smooth_grf]",
                "--set",
                "regimes=[low]",
                "--set",
                "resolution=8",
            ]
        )
        == 0
    )
    generated = capsys.readouterr().out
    assert '"output_root":' in generated
    assert "pdeobs_tiny" in generated
    assert '"num_workers": 1' in generated

    assert (
        main(
            [
                "generate-case",
                "--pde",
                "navier_stokes",
                "--boundary",
                "periodic",
                "--setting",
                "vortex_pair",
                "--param-regime",
                "high",
                "--num-samples",
                "100",
                "--root",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )
    case = capsys.readouterr().out
    assert '"case_id": "navier_stokes/periodic/dipole_vortex_pair/high"' in case
    assert "dipole_vortex_pair" in case
    assert '"tier": "signal"' in case


def test_config_free_train_infer_eval_and_benchmark_preflights(tmp_path: Path, capsys) -> None:
    data = tmp_path / "pdeobs_medium"
    run = tmp_path / "run"
    assert (
        main(
            [
                "train",
                "--task",
                "sparse_recovery",
                "--model",
                "fno",
                "--data",
                str(data),
                "--split",
                "iid",
                "--mask",
                "random_3pct",
                "--output",
                str(run),
                "--dry-run",
            ]
        )
        == 0
    )
    assert '"dry_run": true' in capsys.readouterr().out

    checkpoint = run / "checkpoints" / "best.pt"
    assert (
        main(
            [
                "infer",
                "--task",
                "sparse_recovery",
                "--model",
                "fno",
                "--ckpt",
                str(checkpoint),
                "--data",
                str(data),
                "--split",
                "test",
                "--dry-run",
            ]
        )
        == 0
    )
    infer_payload = capsys.readouterr().out
    assert '"dry_run": true' in infer_payload
    assert "preds.h5" in infer_payload

    assert (
        main(
            [
                "eval",
                "--task",
                "sparse_recovery",
                "--pred",
                str(run / "preds.h5"),
                "--data",
                str(data),
                "--metrics",
                "rel_l2,spectral,pde_residual",
                "--dry-run",
            ]
        )
        == 0
    )
    assert '"dry_run": true' in capsys.readouterr().out

    assert (
        main(
            [
                "benchmark",
                "--preset",
                "fno_sparse_recovery",
                "--tier",
                "medium",
                "--output",
                str(tmp_path / "benchmark"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert '"dry_run": true' in capsys.readouterr().out
