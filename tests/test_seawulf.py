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
    "monitor_full_t15.sh",
    "submit_full_t15.sh",
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
    generation_environment = (ROOT / "environment-generation.yml").read_text(encoding="utf-8")

    assert "PDEOBS_ENV_FILE" in bootstrap
    assert "environment-generation.yml" in bootstrap
    assert "-m pip wheel" in bootstrap
    assert "--force-reinstall --no-deps" in bootstrap
    assert ".pdeobs-git-commit" in bootstrap
    assert ".pdeobs-git-commit" in common
    assert "environment/checkout mismatch" in common
    assert "-e ." not in bootstrap
    assert "-e ." not in environment
    assert "pytorch" not in generation_environment.lower()
    assert "h5py" in generation_environment.lower()


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


def test_download_cli_has_a_manifest_free_release_contract() -> None:
    args = build_parser().parse_args(["download", "--tier", "tiny"])
    assert args.manifest is None


def test_github_readme_exposes_server_and_seawulf_quick_starts() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "[Linux server guide](docs/SERVER.md)" in readme
    assert "[benchmark-paper contract](docs/BENCHMARK_PAPER.md)" in readme
    assert "pdeobs generate-case" in readme
    assert "--preset fno_sparse_recovery" in readme
    assert "runs/<run-id>" not in readme
    assert "## SeaWulf quick start" in readme
    assert '"$PDEOBS_DATA/plans/smoke.jsonl"' in readme
    assert '--dependency="afterok:$smoke_job"' in readme

    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/pdeobs_*/" in ignored
    assert "data/pdeobs_cases/" in ignored


def test_seawulf_guide_uses_exact_plans_and_dependency_chains() -> None:
    guide = (SEAWULF / "README.md").read_text(encoding="utf-8")

    assert "srun --partition=short-40core-shared" in guide
    assert "configs/dataset/recovery_signal.yaml" in guide
    assert '"$PDEOBS_DATA/plans/smoke.jsonl"' in guide
    assert '--dependency="afterok:${generation_job}"' in guide
    assert "does **not** submit model training" in guide
    assert "short-40core-shared" in guide
    assert "about 215 GiB" in guide
    assert '"$PDEOBS_DATA/plans/tiny.resolved.yaml"' in guide
    assert "conservative safety cap" in guide
    assert "numerics_full_t15.yaml" in guide
    assert "0-83%6" in guide
    assert "240 CPU cores" in guide


def test_full_t15_launcher_is_dataset_only_bounded_and_quality_gated() -> None:
    launcher = (SEAWULF / "submit_full_t15.sh").read_text(encoding="utf-8")
    array = (SEAWULF / "generate_array.sbatch").read_text(encoding="utf-8")

    assert 'task_count" != 3360' in launcher
    assert 'sample_count" != 560000' in launcher
    assert "bad_stored_steps" in launcher
    assert "PDEOBS_FULL_CPUS_PER_TASK:-40" in launcher
    assert "PDEOBS_FULL_CONCURRENCY:-6" in launcher
    assert "array_count + 1 > 100" in launcher
    assert "--quality-strict --max-pde-loss 0.05 --require-all-pdes" in launcher
    assert "train_gpu.sbatch" not in launcher
    assert "--array-bundle-size" in array
    assert '"${SLURM_CPUS_PER_TASK:-1}"' in array
    assert "export OMP_NUM_THREADS=1" in array
