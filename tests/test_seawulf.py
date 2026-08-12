from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from pdeobs.cli import build_parser
from pdeobs.config import load_config
from pdeobs.runner import _allow_split_fallback
from pdeobs.splits import build_split_plan, tier_regime_counts

ROOT = Path(__file__).resolve().parents[1]
SEAWULF = ROOT / "hpc" / "seawulf"
SHELL_FILES = (
    "aggregate_cpu.sbatch",
    "bootstrap.sh",
    "common.sh",
    "evaluate_gpu.sbatch",
    "generate_array.sbatch",
    "submit_generation.sh",
    "train_gpu.sbatch",
)


def _bash_executable() -> str | None:
    discovered = shutil.which("bash")
    if discovered:
        return discovered
    if os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/bin/bash.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def test_seawulf_shell_files_parse_without_a_slurm_installation() -> None:
    bash = _bash_executable()
    if bash is None:
        pytest.skip("bash is unavailable")
    relative_paths = [
        path.relative_to(ROOT).as_posix() for path in map(SEAWULF.__truediv__, SHELL_FILES)
    ]
    subprocess.run([bash, "-n", *relative_paths], cwd=ROOT, check=True)


def test_seawulf_runtime_restores_slurm_after_module_purge() -> None:
    for name in ("bootstrap.sh", "common.sh"):
        script = (SEAWULF / name).read_text(encoding="utf-8")
        purge = script.index("module purge")
        slurm = script.index("module load slurm", purge)
        anaconda = script.index("module load anaconda/3", slurm)
        assert purge < slurm < anaconda


def test_seawulf_environment_is_wheel_installed_and_commit_guarded() -> None:
    bootstrap = (SEAWULF / "bootstrap.sh").read_text(encoding="utf-8")
    common = (SEAWULF / "common.sh").read_text(encoding="utf-8")
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")

    assert "-m pip wheel" in bootstrap
    assert "--force-reinstall --no-deps" in bootstrap
    assert ".pdeobs-git-commit" in bootstrap
    assert ".pdeobs-git-commit" in common
    assert "environment/checkout mismatch" in common
    assert "-e ." not in bootstrap
    assert "-e ." not in environment


def test_cluster_yaml_is_explicitly_documentation_only() -> None:
    config = yaml.safe_load((ROOT / "configs/cluster/seawulf.yaml").read_text(encoding="utf-8"))
    assert config["documentation_only"] is True


def test_production_and_smoke_split_policies_are_explicit() -> None:
    production = load_config(ROOT / "configs/experiment/recovery_unet.yaml")
    smoke = load_config(ROOT / "configs/experiment/recovery_unet_smoke.yaml")
    transparent_smoke = load_config(ROOT / "configs/experiment/recovery_nearest_smoke.yaml")

    assert production["data"]["verify_shards"] is True
    assert _allow_split_fallback(production) is False
    assert production["data"]["root"].endswith("/signal")
    for config in (smoke, transparent_smoke):
        assert config["data"]["verify_shards"] is True
        assert _allow_split_fallback(config) is True


def test_focused_recovery_signal_case_contains_every_iid_split() -> None:
    config = load_config(ROOT / "configs/dataset/recovery_signal.yaml")
    count = tier_regime_counts(config["tier"])["low"]
    plan = build_split_plan(
        2000,
        seed=config["seed"],
        case_key="poisson/periodic/smooth_grf",
    )

    assert {plan[index].split for index in range(count)} == {
        "train",
        "validation",
        "test",
    }


def test_split_fallback_flag_rejects_non_boolean_values() -> None:
    with pytest.raises(ValueError, match="allow_split_fallback"):
        _allow_split_fallback({"data": {"allow_split_fallback": "yes"}})


def test_download_requires_an_explicit_release_manifest() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["download", "--tier", "tiny"])
