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
SLURM = ROOT / "hpc" / "slurm"
SHELL_FILES = (
    "aggregate_cpu.sbatch",
    "bootstrap.sh",
    "common.sh",
    "evaluate_gpu.sbatch",
    "generate_array.sbatch",
    "monitor_full_t15.sh",
    "submit_full_t15.sh",
    "submit_generation.sh",
    "submit_validation20.sh",
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


def test_slurm_shell_files_parse_without_a_slurm_installation() -> None:
    bash = _bash_executable()
    if bash is None:
        pytest.skip("bash is unavailable")
    relative_paths = [
        path.relative_to(ROOT).as_posix() for path in map(SLURM.__truediv__, SHELL_FILES)
    ]
    subprocess.run([bash, "-n", *relative_paths], cwd=ROOT, check=True)


def test_slurm_runtime_has_no_site_specific_software_assumptions() -> None:
    scripts = "\n".join(
        (SLURM / name).read_text(encoding="utf-8") for name in ("bootstrap.sh", "common.sh")
    )
    assert "PDEOBS_MODULES" in scripts
    assert "PDEOBS_MODULE_SETUP" in scripts
    assert "PDEOBS_ENV_MANAGER" in scripts
    assert "module load slurm" not in scripts
    assert "module load anaconda" not in scripts
    assert "module load cuda" not in scripts


def test_slurm_environment_is_wheel_installed_and_commit_guarded() -> None:
    bootstrap = (SLURM / "bootstrap.sh").read_text(encoding="utf-8")
    common = (SLURM / "common.sh").read_text(encoding="utf-8")
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


def test_cluster_yaml_is_documentation_only_and_portable() -> None:
    config = yaml.safe_load((ROOT / "configs/cluster/slurm.yaml").read_text(encoding="utf-8"))
    assert config["documentation_only"] is True
    assert config["cluster"] == "slurm"
    assert config["routing"]["cpu_partition_env"] == "PDEOBS_CPU_PARTITION"
    assert config["routing"]["gpu_partition_env"] == "PDEOBS_GPU_PARTITION"
    assert "partition" not in config["examples"]["cpu"]
    assert "partition" not in config["examples"]["gpu"]


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


def test_github_readme_exposes_server_and_slurm_quick_starts() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "[Linux server guide](docs/SERVER.md)" in readme
    assert "[benchmark-paper contract](docs/BENCHMARK_PAPER.md)" in readme
    assert "pdeobs generate-case" in readme
    assert "--preset fno_sparse_recovery" in readme
    assert "runs/<run-id>" not in readme
    assert "## Slurm HPC quick start" in readme
    assert '"$PDEOBS_DATA/plans/smoke.jsonl"' in readme
    assert '--dependency="afterok:$smoke_job"' in readme

    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/pdeobs_*/" in ignored
    assert "data/pdeobs_cases/" in ignored


def test_slurm_guide_uses_exact_plans_and_dependency_chains() -> None:
    guide = (SLURM / "README.md").read_text(encoding="utf-8")

    assert "portable Slurm templates" in guide
    assert "do not name a cluster, partition, account, QOS" in guide
    assert '"$PDEOBS_DATA/plans/smoke.jsonl"' in guide
    assert '--dependency="afterok:${generation_job}"' in guide
    assert "does **not** submit model training" in guide
    assert "PDEOBS_CPU_PARTITION" in guide
    assert "PDEOBS_GPU_PARTITION" in guide
    assert "PDEOBS_ACCOUNT" in guide
    assert "PDEOBS_MAX_QUEUED_TASKS" in guide
    assert "Slurm documentation" in guide


def test_full_t15_launcher_is_dataset_only_bounded_and_quality_gated() -> None:
    launcher = (SLURM / "submit_full_t15.sh").read_text(encoding="utf-8")
    array = (SLURM / "generate_array.sbatch").read_text(encoding="utf-8")

    assert 'task_count" != 3360' in launcher
    assert 'sample_count" != 560000' in launcher
    assert "bad_stored_steps" in launcher
    assert "PDEOBS_FULL_CPUS_PER_TASK:-8" in launcher
    assert "PDEOBS_FULL_CONCURRENCY:-4" in launcher
    assert "PDEOBS_MAX_QUEUED_TASKS:-100" in launcher
    assert "array_count + 1 > max_queued" in launcher
    assert "PDEOBS_CPU_PARTITION" in launcher
    assert "PDEOBS_ACCOUNT" in launcher
    assert "--quality-strict --max-pde-loss 0.05 --require-all-pdes" in launcher
    assert "train_gpu.sbatch" not in launcher
    assert "--array-bundle-size" in array
    assert '"${SLURM_CPUS_PER_TASK:-1}"' in array
    assert "export OMP_NUM_THREADS=1" in array


def test_slurm_templates_do_not_pin_site_resources() -> None:
    text = "\n".join((SLURM / name).read_text(encoding="utf-8") for name in SHELL_FILES)
    forbidden = (
        "#SBATCH --partition=",
        "#SBATCH --account=",
        "#SBATCH --qos=",
        "#SBATCH --constraint=",
    )
    for token in forbidden:
        assert token not in text
