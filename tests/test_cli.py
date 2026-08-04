from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pdeobs.cli import main


def test_cli_lists_components(capsys) -> None:
    assert main(["list", "--kind", "pdes", "--json"]) == 0
    output = capsys.readouterr().out
    assert '"poisson"' in output
    assert '"navier_stokes"' in output


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
