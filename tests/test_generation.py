import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pdeobs.aggregate import validate_hdf5_shard
from pdeobs.generation import (
    build_job_grid,
    generate_job,
    jobs_from_spec,
    load_job_manifest,
    run_generation,
    select_array_job,
    write_generation_plan,
    write_job_manifest,
)
from pdeobs.quality import QUALITY_SCHEMA_VERSION, QualityGateError
from pdeobs.registry import PDE_REGISTRY
from pdeobs.schema import GenerationSpec, Sample
from pdeobs.storage import (
    AtomicHDF5ShardWriter,
    IncompleteShardError,
    LazyHDF5Dataset,
    StorageError,
    is_shard_complete,
    read_shard_manifest,
)


def test_job_grid_balances_tiny_tier_and_manifest_round_trip(tmp_path, monkeypatch):
    jobs = build_job_grid(
        tmp_path,
        tier="tiny",
        resolution=8,
        shard_size=2,
        families=("poisson",),
        boundaries=("periodic",),
        settings=("smooth_grf",),
    )
    assert len(jobs) == 3
    assert sum(job.sample_count for job in jobs) == 5
    manifest = write_job_manifest(jobs, tmp_path / "jobs.jsonl")
    assert load_job_manifest(manifest) == jobs
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    assert select_array_job(manifest).job_id == jobs[2].job_id


def test_job_grid_canonicalizes_setting_aliases_in_ids_and_paths(tmp_path):
    job = build_job_grid(
        tmp_path,
        tier="tiny",
        families=("poisson",),
        boundaries=("periodic",),
        settings=("dipole",),
        regimes=("low",),
    )[0]
    assert job.setting == "dipole_vortex_pair"
    assert "dipole_vortex_pair" in Path(job.output_path).parts
    assert "dipole_vortex_pair" in job.case_key.split("/")


def test_job_grid_supports_family_specific_saved_time_resolution(tmp_path):
    jobs = build_job_grid(
        tmp_path,
        tier="tiny",
        time_steps=9,
        time_steps_by_family={"reaction-diffusion": 65, "burgers": 129},
        families=("heat", "reaction_diffusion", "burgers"),
        boundaries=("periodic",),
        settings=("smooth_grf",),
        regimes=("low",),
    )

    assert {job.pde: job.time_steps for job in jobs} == {
        "heat": 9,
        "reaction_diffusion": 65,
        "burgers": 129,
    }


def test_job_grid_supports_case_specific_solver_and_saved_frames(tmp_path):
    jobs = build_job_grid(
        tmp_path,
        tier="tiny",
        time_steps=9,
        time_steps_by_case={
            "navier_stokes/periodic": 33,
            "navier_stokes/periodic/multi_frequency_fourier": 65,
            "navier_stokes/robin_obstacle": 257,
        },
        options={"shared": True},
        options_by_case={
            "navier_stokes/periodic": {"solver": "fno_spectral_vorticity"},
            "navier_stokes/robin_obstacle": {"solver": "masked_obstacle_vorticity"},
        },
        families=("navier_stokes",),
        boundaries=("periodic", "robin_obstacle"),
        settings=("smooth_grf", "multi_frequency_fourier"),
        regimes=("low",),
    )

    by_case = {(job.boundary, job.setting): job for job in jobs}
    assert by_case[("periodic", "smooth_grf")].time_steps == 33
    assert by_case[("periodic", "multi_frequency_fourier")].time_steps == 65
    assert by_case[("robin_obstacle", "smooth_grf")].time_steps == 257
    assert by_case[("periodic", "smooth_grf")].options == {
        "shared": True,
        "solver": "fno_spectral_vorticity",
    }
    assert by_case[("robin_obstacle", "smooth_grf")].options == {
        "shared": True,
        "solver": "masked_obstacle_vorticity",
    }


@pytest.mark.parametrize("value", (0, -1))
def test_job_grid_rejects_invalid_family_time_steps(tmp_path, value):
    with pytest.raises(ValueError, match="must be positive"):
        build_job_grid(
            tmp_path,
            tier="tiny",
            time_steps_by_family={"burgers": value},
            families=("burgers",),
            boundaries=("periodic",),
            settings=("smooth_grf",),
            regimes=("low",),
        )


def test_job_grid_rejects_unregistered_setting_path(tmp_path):
    with pytest.raises(ValueError, match="unknown setting"):
        build_job_grid(
            tmp_path,
            tier="tiny",
            families=("poisson",),
            boundaries=("periodic",),
            settings=("../../escaped",),
            regimes=("low",),
        )


def test_generation_rejects_non_floating_storage_dtype(tmp_path):
    with pytest.raises(TypeError, match="floating-point"):
        build_job_grid(
            tmp_path,
            tier="tiny",
            dtype="int8",
            families=("poisson",),
            boundaries=("periodic",),
            settings=("smooth_grf",),
            regimes=("low",),
        )


def test_job_grid_rejects_unknown_pde_after_plugin_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr("pdeobs.pdes._PLUGINS_DISCOVERED", False)
    monkeypatch.setattr(PDE_REGISTRY, "discover", lambda **_: ())
    with pytest.raises(ValueError, match="unknown PDE family"):
        build_job_grid(
            tmp_path,
            tier="tiny",
            families=("../../not-a-solver",),
            boundaries=("periodic",),
            settings=("smooth_grf",),
            regimes=("low",),
        )


def test_external_pde_plugin_is_discovered_and_generated(tmp_path, monkeypatch):
    monkeypatch.setattr("pdeobs.pdes._PLUGINS_DISCOVERED", False)
    monkeypatch.setattr(PDE_REGISTRY, "_objects", dict(PDE_REGISTRY._objects))
    monkeypatch.setattr(PDE_REGISTRY, "_aliases", dict(PDE_REGISTRY._aliases))
    discovery_calls = 0
    received_time_steps: list[int | None] = []

    def external_solver(
        boundary="periodic",
        setting="smooth_grf",
        regime="low",
        seed=0,
        resolution=(8, 8),
        time_steps=None,
        **_options,
    ):
        del boundary, setting, regime, seed
        received_time_steps.append(time_steps)
        height, width = resolution
        steps = int(time_steps or 1)
        condition = np.ones((height, width, 3), dtype=np.float32)
        trajectory = np.ones((steps, height, width, 4), dtype=np.float32)
        geometry = np.zeros((height, width, 1), dtype=np.bool_)
        return Sample(
            condition,
            trajectory,
            geometry,
            {"plugin": True, "state_representation": "plugin_vector"},
        )

    def discover(**_kwargs):
        nonlocal discovery_calls
        discovery_calls += 1
        PDE_REGISTRY.register("external_wave", external_solver)
        return ("external_wave",)

    monkeypatch.setattr(PDE_REGISTRY, "discover", discover)
    job = build_job_grid(
        tmp_path,
        tier="tiny",
        resolution=8,
        shard_size=2,
        time_steps=4,
        families=("external-wave",),
        boundaries=("periodic",),
        settings=("smooth_grf",),
        regimes=("low",),
    )[0]
    assert job.pde == "external_wave"
    assert "external_wave" in Path(job.output_path).parts
    assert discovery_calls == 1

    result = generate_job(job)
    assert result.sample_count == 2
    assert received_time_steps == [4, 4]
    with LazyHDF5Dataset(job.output_path) as dataset:
        assert dataset[0].condition.shape == (8, 8, 3)
        assert dataset[0].trajectory.shape == (4, 8, 8, 4)
        assert dataset[0].geometry.dtype == np.bool_
        assert dataset[0].metadata["pde"] == "external_wave"
        assert dataset[0].metadata["state_representation"] == "plugin_vector"
        assert dataset[0].metadata["solver_fidelity"] == "external_plugin"
        assert dataset[0].metadata["solver_version"] == "unreported"
        assert dataset[0].metadata["solver_implementation"].endswith("external_solver")
        assert dataset[0].metadata["quality"]["pde_loss"]["status"] == "unsupported"
        assert dataset[0].metadata["quality"]["checks"]["array_contract"] == "pass"
        assert dataset[0].metadata["quality"]["operator_id"] is None

    validate_hdf5_shard(job.output_path, verify_checksum=True, strict=True)


def test_job_grid_rejects_duplicate_setting_aliases(tmp_path):
    with pytest.raises(ValueError, match="same canonical setting"):
        build_job_grid(
            tmp_path,
            tier="tiny",
            families=("poisson",),
            boundaries=("periodic",),
            settings=("dipole", "dipole_vortex_pair"),
            regimes=("low",),
        )


@pytest.mark.parametrize(
    ("key", "values"),
    (
        ("families", ("poisson", "poisson")),
        ("boundaries", ("periodic", "periodic")),
        ("regimes", ("low", "low")),
    ),
)
def test_job_grid_rejects_duplicate_factors(tmp_path, key, values):
    kwargs = {
        "families": ("poisson",),
        "boundaries": ("periodic",),
        "settings": ("smooth_grf",),
        "regimes": ("low",),
    }
    kwargs[key] = values
    with pytest.raises(ValueError, match="duplicate"):
        build_job_grid(tmp_path, tier="tiny", **kwargs)


def test_jobs_from_spec_uses_canonical_setting_path(tmp_path):
    spec = GenerationSpec(
        pde="poisson",
        boundary="periodic",
        setting="dipole",
        regime="low",
        num_samples=1,
    )
    job = jobs_from_spec(spec, tmp_path)[0]
    assert job.setting == "dipole_vortex_pair"
    assert "dipole_vortex_pair" in Path(job.output_path).parts


def test_config_adapters_write_plan_and_produce_json_safe_dry_run(tmp_path):
    config = {
        "tier": "tiny",
        "resolution": 8,
        "trajectory_steps": 9,
        "shard_size": 2,
        "families": ["poisson"],
        "boundaries": ["periodic"],
        "settings": ["smooth_grf"],
        "regimes": ["low"],
        "output": {"root": str(tmp_path / "configured")},
    }
    plan_path = tmp_path / "plan.jsonl"
    planned = write_generation_plan(config, plan_path)
    assert len(planned) == 1
    summary = run_generation(config, tmp_path / "cli-output", plan_path=plan_path, dry_run=True)
    assert summary["status"] == "dry_run"
    assert summary["sample_count"] == 2
    assert Path(summary["jobs"][0]["output_path"]).is_relative_to(tmp_path / "cli-output")


def test_plan_rejects_a_different_runtime_code_identity(tmp_path):
    planned_config = {
        "tier": "tiny",
        "families": ["poisson"],
        "boundaries": ["periodic"],
        "settings": ["smooth_grf"],
        "regimes": ["low"],
        "_provenance": {
            "config_hash": "config-a",
            "git": {"commit": "commit-a", "dirty": False, "status": ""},
        },
    }
    plan = tmp_path / "identity.jsonl"
    write_generation_plan(planned_config, plan)
    current_config = {
        **planned_config,
        "_provenance": {
            "config_hash": "config-a",
            "git": {"commit": "commit-b", "dirty": False, "status": ""},
        },
    }
    with pytest.raises(ValueError, match="current checkout"):
        run_generation(current_config, tmp_path / "output", plan_path=plan, dry_run=True)


def test_atomic_writer_resumes_partial_and_lazy_reader(tmp_path):
    pytest.importorskip("h5py")
    path = tmp_path / "resume.h5"

    def sample(index):
        return Sample(
            np.full((6, 5), index, dtype=np.float32),
            np.full((1, 6, 5), index + 1, dtype=np.float32),
            np.zeros((6, 5), dtype=np.float32),
            {"index": index},
        )

    first = AtomicHDF5ShardWriter(path, expected_count=2, spec={"case": "test"})
    first.append(sample(0))
    first.close()
    resumed = AtomicHDF5ShardWriter(path, expected_count=2, spec={"case": "test"})
    assert resumed.count == 1
    resumed.append(sample(1))
    resumed.finalize()

    assert is_shard_complete(path, expected_count=2)
    assert path.with_suffix(".manifest.json").exists()
    assert path.with_suffix(".metadata.csv").exists()
    assert path.with_suffix(".metadata.json").exists()
    with LazyHDF5Dataset(path, verify=True) as dataset:
        assert len(dataset) == 2
        assert dataset[1].metadata["index"] == 1
        assert dataset[1].trajectory.shape == (1, 6, 5, 1)


def test_atomic_writer_excludes_concurrent_shard_writers(tmp_path):
    pytest.importorskip("h5py")
    path = tmp_path / "exclusive.h5"
    writer = AtomicHDF5ShardWriter(path, expected_count=1, spec={"case": "lock"})
    try:
        assert path.with_name(path.name + ".lock").is_file()
        with pytest.raises(StorageError, match="already locked"):
            AtomicHDF5ShardWriter(path, expected_count=1, spec={"case": "lock"})
    finally:
        writer.close()
    assert not path.with_name(path.name + ".lock").exists()


def test_atomic_writer_rejects_completed_or_partial_shards_from_another_spec(tmp_path):
    pytest.importorskip("h5py")

    def sample(index):
        return Sample(
            np.full((6, 5), index, dtype=np.float32),
            np.full((1, 6, 5), index + 1, dtype=np.float32),
            np.zeros((6, 5), dtype=np.float32),
            {"index": index},
        )

    completed_path = tmp_path / "completed.h5"
    original_spec = {
        "seed": 1,
        "options": {"modes": [4, 8]},
        "output_path": "/scratch/first/shard.h5",
        "job_id": "first-audit-id",
        "provenance": {
            "captured_at_utc": "2026-01-01T00:00:00+00:00",
            "config_hash": "config-a",
            "source_hash": "source-a",
            "git": {"commit": "commit-a", "dirty": False, "status": ""},
            "runtime": {
                "executable": "/first/python",
                "hostname": "node-a",
                "python": "3.12.1",
            },
            "slurm": {"SLURM_JOB_ID": "101"},
        },
    }
    with AtomicHDF5ShardWriter(completed_path, expected_count=1, spec=original_spec) as writer:
        writer.append(sample(0))

    reordered_spec = {
        "provenance": {
            "captured_at_utc": "2026-01-02T00:00:00+00:00",
            "config_hash": "config-a",
            "source_hash": "source-a",
            "git": {"commit": "commit-a", "dirty": False, "status": ""},
            "runtime": {
                "executable": "/second/python",
                "hostname": "node-b",
                "python": "3.12.1",
            },
            "slurm": {"SLURM_JOB_ID": "202"},
        },
        "job_id": "second-audit-id",
        "output_path": "/scratch/second/shard.h5",
        "options": {"modes": [4, 8]},
        "seed": 1,
    }
    matching = AtomicHDF5ShardWriter(completed_path, expected_count=1, spec=reordered_spec)
    assert matching.completed
    status_only_change = {
        **reordered_spec,
        "provenance": {
            **reordered_spec["provenance"],
            "git": {
                "commit": "commit-a",
                "dirty": True,
                "status": "?? data/pdeobs_tiny/_generation/resolved.yaml",
            },
        },
    }
    assert AtomicHDF5ShardWriter(
        completed_path, expected_count=1, spec=status_only_change
    ).completed
    with pytest.raises(IncompleteShardError, match="different job spec"):
        AtomicHDF5ShardWriter(
            completed_path,
            expected_count=1,
            spec={**reordered_spec, "seed": 2},
        )
    changed_commit = {
        **reordered_spec,
        "provenance": {
            **reordered_spec["provenance"],
            "git": {"commit": "commit-b", "dirty": False, "status": ""},
        },
    }
    with pytest.raises(IncompleteShardError, match="different job spec"):
        AtomicHDF5ShardWriter(
            completed_path,
            expected_count=1,
            spec=changed_commit,
        )
    changed_config = {
        **reordered_spec,
        "provenance": {**reordered_spec["provenance"], "config_hash": "config-b"},
    }
    with pytest.raises(IncompleteShardError, match="different job spec"):
        AtomicHDF5ShardWriter(
            completed_path,
            expected_count=1,
            spec=changed_config,
        )
    changed_source = {
        **reordered_spec,
        "provenance": {**reordered_spec["provenance"], "source_hash": "source-b"},
    }
    with pytest.raises(IncompleteShardError, match="different job spec"):
        AtomicHDF5ShardWriter(
            completed_path,
            expected_count=1,
            spec=changed_source,
        )

    partial_path = tmp_path / "partial.h5"
    partial = AtomicHDF5ShardWriter(partial_path, expected_count=2, spec=original_spec)
    partial.append(sample(0))
    partial.close()
    resumed = AtomicHDF5ShardWriter(partial_path, expected_count=2, spec=reordered_spec)
    assert resumed.count == 1
    resumed.append(sample(1))
    resumed.finalize()
    assert read_shard_manifest(partial_path)["spec"] == original_spec

    mismatched_partial_path = tmp_path / "mismatched-partial.h5"
    mismatched = AtomicHDF5ShardWriter(
        mismatched_partial_path, expected_count=2, spec=original_spec
    )
    mismatched.append(sample(0))
    mismatched.close()
    with pytest.raises(IncompleteShardError, match="different job spec"):
        AtomicHDF5ShardWriter(
            mismatched_partial_path,
            expected_count=2,
            spec={**reordered_spec, "seed": 2},
        )


def test_generate_job_skips_identical_content_with_fresh_provenance(tmp_path):
    pytest.importorskip("h5py")
    job = build_job_grid(
        tmp_path,
        tier="tiny",
        resolution=8,
        shard_size=2,
        families=("poisson",),
        boundaries=("periodic",),
        settings=("smooth_grf",),
        regimes=("low",),
        provenance={
            "captured_at_utc": "2026-01-01T00:00:00+00:00",
            "config_hash": "config-a",
            "git": {"commit": "commit-a", "dirty": False, "status": ""},
            "runtime": {
                "executable": "/login/python",
                "hostname": "login-a",
                "python": "3.12.1",
            },
            "slurm": {"SLURM_JOB_ID": "101"},
        },
    )[0]
    first = generate_job(job)
    rerun = generate_job(
        replace(
            job,
            provenance={
                "captured_at_utc": "2026-01-02T00:00:00+00:00",
                "config_hash": "config-a",
                "git": {"commit": "commit-a", "dirty": False, "status": ""},
                "runtime": {
                    "executable": "/compute/python",
                    "hostname": "compute-b",
                    "python": "3.12.1",
                },
                "slurm": {"SLURM_JOB_ID": "202"},
            },
        )
    )
    assert not first.skipped
    assert rerun.skipped
    assert rerun.sha256 == first.sha256


def test_small_generation_job_is_deterministic_and_resumable(tmp_path):
    pytest.importorskip("h5py")
    job = build_job_grid(
        tmp_path,
        tier="tiny",
        resolution=8,
        shard_size=2,
        families=("heat",),
        boundaries=("periodic",),
        settings=("smooth_grf",),
        regimes=("low",),
        seed=19,
    )[0]
    first = generate_job(job)
    second = generate_job(job)
    assert not first.skipped
    assert second.skipped
    assert first.sha256 == second.sha256
    with LazyHDF5Dataset(job.output_path) as dataset:
        assert len(dataset) == 2
        assert dataset[0].trajectory.shape == (9, 8, 8, 1)
        metadata = dataset[0].metadata
        assert metadata["T"] == 9
        assert metadata["sample_id"].startswith("seed-19/")
        assert metadata["boundary_ood"] is False
        assert metadata["parameter_ood"] is False
        assert metadata["solver_fidelity"] == "compact_reference"
        assert metadata["solver_version"] == "0.1.0"
        assert metadata["solver_implementation"].startswith("pdeobs.pdes.heat:")
        quality = metadata["quality"]
        assert quality["schema_version"] == QUALITY_SCHEMA_VERSION
        assert quality["pde"] == "heat"
        assert quality["stored_dtype"] == "float32"
        assert quality["pde_loss"]["status"] == "measured"
        assert quality["metrics"]["pde_loss_normalized"] >= 0.0
        assert quality["status"] == "warning"


def test_dense_quality_trajectory_stores_exact_uniform_30_frame_subset(tmp_path):
    pytest.importorskip("h5py")
    job = build_job_grid(
        tmp_path,
        tier="tiny",
        resolution=8,
        shard_size=2,
        families=("heat",),
        boundaries=("periodic",),
        settings=("smooth_grf",),
        regimes=("low",),
        seed=23,
        time_steps=59,
        stored_time_steps=30,
    )[0]
    generate_job(job)
    with LazyHDF5Dataset(job.output_path) as dataset:
        sample = dataset[0]
        metadata = sample.metadata
        assert sample.trajectory.shape == (30, 8, 8, 1)
        assert metadata["T"] == 30
        assert metadata["quality_T"] == 59
        assert metadata["quality_source"] == "dense_pre_storage_trajectory"
        assert metadata["stored_frame_indices"] == list(range(0, 59, 2))
        assert metadata["quality"]["calibration_context"]["T"] == 59
        assert len(metadata["stored_time_values"]) == 30
    validate_hdf5_shard(job.output_path, verify_checksum=True, strict=True)


def test_stored_time_axis_rejects_interpolation_or_upsampling(tmp_path):
    with pytest.raises(ValueError, match="divisible"):
        build_job_grid(
            tmp_path,
            tier="tiny",
            families=("heat",),
            boundaries=("periodic",),
            settings=("smooth_grf",),
            regimes=("low",),
            time_steps=58,
            stored_time_steps=30,
        )


def test_quality_describes_exact_float64_storage_and_rejected_sample_is_audited(tmp_path):
    pytest.importorskip("h5py")
    job = build_job_grid(
        tmp_path / "float64",
        tier="tiny",
        resolution=8,
        shard_size=2,
        dtype="float64",
        families=("poisson",),
        boundaries=("periodic",),
        settings=("smooth_grf",),
        regimes=("low",),
    )[0]
    generate_job(job)
    with LazyHDF5Dataset(job.output_path) as dataset:
        assert dataset[0].trajectory.dtype == np.dtype("float64")
        assert dataset[0].metadata["quality"]["stored_dtype"] == "float64"

    rejected = build_job_grid(
        tmp_path / "rejected",
        tier="tiny",
        resolution=8,
        shard_size=2,
        families=("poisson",),
        boundaries=("periodic",),
        settings=("smooth_grf",),
        regimes=("low",),
        quality={
            "profile": "strict",
            "thresholds": {"pde_loss_normalized_max": 0.0},
        },
    )[0]
    with pytest.raises(QualityGateError):
        generate_job(rejected)
    failure_path = Path(rejected.output_path).with_suffix(".quality-failures.jsonl")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["accepted"] is False
    assert failure["quality"]["status"] == "fail"
    assert failure["quality"]["checks"]["pde_loss"] == "fail"
