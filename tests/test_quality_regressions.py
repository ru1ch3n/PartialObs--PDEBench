from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from pdeobs.aggregate import ShardValidationError, validate_hdf5_shard
from pdeobs.pdes import PDE_FAMILIES, STATIC_FAMILIES, generate_sample
from pdeobs.quality import (
    QualityGateError,
    audit_dataset_quality,
    enforce_generation_quality,
    evaluate_sample_quality,
    generation_quality_rejected,
)
from pdeobs.schema import Sample
from pdeobs.storage import AtomicHDF5ShardWriter, is_shard_complete, shard_sidecars


def _generated_sample(family: str, *, boundary: str = "periodic") -> Sample:
    output = generate_sample(
        family,
        boundary=boundary,
        setting="smooth_grf",
        regime="low",
        seed=31,
        resolution=8,
        time_steps=None if family in STATIC_FAMILIES else 4,
    )
    return Sample(
        output.condition,
        output.trajectory,
        output.geometry,
        {
            "pde": output.family,
            "boundary": output.boundary,
            "setting": output.setting,
            "regime": output.regime,
            "parameters": dict(output.parameters),
            "solver_fidelity": "compact_reference",
        },
    )


@pytest.mark.parametrize("family", PDE_FAMILIES)
def test_every_builtin_family_generates_robin_data(family: str) -> None:
    output = generate_sample(
        family,
        boundary="robin",
        setting="smooth_grf",
        regime="medium",
        seed=23,
        resolution=(8, 10),
        time_steps=None if family in STATIC_FAMILIES else 4,
    )

    expected_steps = 1 if family in STATIC_FAMILIES else 4
    assert output.boundary == "robin"
    assert output.condition.shape[:2] == (8, 10)
    assert output.trajectory.shape[:3] == (expected_steps, 8, 10)
    assert output.geometry.shape == (8, 10, 1)
    assert all(
        np.all(np.isfinite(array))
        for array in (output.condition, output.trajectory, output.geometry)
    )


@pytest.mark.parametrize("family", PDE_FAMILIES)
def test_missing_required_domain_contract_is_unsupported_and_fails(family: str) -> None:
    sample = _generated_sample(family)
    parameters = dict(sample.metadata["parameters"])
    parameters.pop("domain_id")
    malformed = Sample(
        sample.condition,
        sample.trajectory,
        sample.geometry,
        {**sample.metadata, "parameters": parameters},
    )

    report = evaluate_sample_quality(malformed)

    assert report["pde_loss"]["available"] is False
    assert report["pde_loss"]["status"] == "unsupported"
    assert "domain_id" in report["pde_loss"]["reason"]
    assert report["checks"]["pde_loss"] == "fail"
    assert report["status"] == "fail"
    json.dumps(report, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize(
    ("family", "parameter"),
    (
        ("darcy", "forcing_amplitude"),
        ("helmholtz", "wavenumber"),
        ("heat", "diffusivity"),
        ("reaction_diffusion", "reaction_rate"),
        ("burgers", "viscosity"),
        ("navier_stokes", "viscosity"),
    ),
)
def test_nonfinite_required_equation_parameter_is_unsupported_and_never_emitted(
    family: str, parameter: str
) -> None:
    sample = _generated_sample(family)
    parameters = dict(sample.metadata["parameters"])
    assert parameter in parameters
    parameters[parameter] = float("nan")
    malformed = Sample(
        sample.condition,
        sample.trajectory,
        sample.geometry,
        {**sample.metadata, "parameters": parameters},
    )

    report = evaluate_sample_quality(malformed)

    assert report["pde_loss"]["available"] is False
    assert report["pde_loss"]["status"] == "unsupported"
    assert parameter in report["pde_loss"]["reason"]
    assert report["checks"]["pde_loss"] == "fail"
    assert report["status"] == "fail"
    json.dumps(report, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize(
    ("family", "parameter"),
    (
        ("darcy", "forcing_amplitude"),
        ("darcy", "solver_steps"),
        ("poisson", "solver_steps"),
        ("helmholtz", "wavenumber"),
        ("helmholtz", "damping_ratio"),
        ("heat", "diffusivity"),
        ("reaction_diffusion", "reaction_rate"),
        ("burgers", "viscosity"),
        ("navier_stokes", "viscosity"),
    ),
)
def test_calibration_key_changes_with_equation_parameter(family: str, parameter: str) -> None:
    sample = _generated_sample(family)
    baseline = evaluate_sample_quality(sample)
    parameters = dict(sample.metadata["parameters"])
    original = float(parameters[parameter])
    parameters[parameter] = original * 1.25 + 0.125
    changed = Sample(
        sample.condition,
        sample.trajectory,
        sample.geometry,
        {**sample.metadata, "parameters": parameters},
    )

    changed_report = evaluate_sample_quality(changed)

    assert changed_report["calibration_key"] != baseline["calibration_key"]


def test_extra_channel_builtin_sample_fails_array_contract() -> None:
    sample = _generated_sample("poisson")
    extra_channel_sample = Sample(
        np.repeat(sample.condition, 2, axis=-1),
        np.repeat(sample.trajectory, 2, axis=-1),
        sample.geometry,
        sample.metadata,
    )

    report = evaluate_sample_quality(extra_channel_sample)

    assert report["checks"]["array_contract"] == "fail"
    assert report["pde_loss"]["available"] is False
    assert report["pde_loss"]["status"] == "unsupported"
    assert report["status"] == "fail"
    json.dumps(report, allow_nan=False, sort_keys=True)


def test_external_multichannel_sample_keeps_generic_contract_reportable() -> None:
    sample = Sample(
        np.zeros((6, 7, 3), dtype=np.float32),
        np.zeros((2, 6, 7, 4), dtype=np.float32),
        np.zeros((6, 7, 1), dtype=np.bool_),
        {
            "pde": "external_vector_system",
            "boundary": "periodic",
            "setting": "smooth_grf",
            "regime": "low",
            "state_representation": "plugin_vector",
            "parameters": {},
        },
    )

    report = evaluate_sample_quality(sample)

    assert report["checks"]["array_contract"] == "pass"
    assert report["operator_id"] is None
    assert report["pde_loss"]["available"] is False
    assert generation_quality_rejected(report) is False
    json.dumps(report, allow_nan=False, sort_keys=True)


def test_report_mode_quarantines_builtin_with_missing_temporal_residual() -> None:
    sample = _generated_sample("heat")
    parameters = dict(sample.metadata["parameters"])
    parameters["final_time"] = 0.0
    malformed = Sample(
        sample.condition,
        sample.trajectory,
        sample.geometry,
        {**sample.metadata, "parameters": parameters},
    )

    report = evaluate_sample_quality(malformed, config={"profile": "report"})

    assert report["checks"]["pde_residual_contract"] == "fail"
    assert generation_quality_rejected(report) is True
    with pytest.raises(QualityGateError, match="pde_residual_contract"):
        enforce_generation_quality(report)


def test_nonfinite_bounded_navier_stokes_parameter_remains_unsupported() -> None:
    sample = _generated_sample("navier_stokes", boundary="robin")
    parameters = dict(sample.metadata["parameters"])
    parameters["viscosity"] = float("nan")
    malformed = Sample(
        sample.condition,
        sample.trajectory,
        sample.geometry,
        {**sample.metadata, "parameters": parameters},
    )

    report = evaluate_sample_quality(malformed)

    assert sample.trajectory.shape[-1] == 2
    assert report["pde_loss"]["available"] is False
    assert report["pde_loss"]["status"] == "unsupported"
    assert "viscosity" in report["pde_loss"]["reason"]
    assert report["checks"]["pde_loss"] == "fail"
    assert report["status"] == "fail"
    json.dumps(report, allow_nan=False, sort_keys=True)


def _write_contract_shard(path: Path) -> Path:
    spec = {
        "pde": "poisson",
        "boundary": "periodic",
        "setting": "smooth_grf",
        "regime": "low",
        "sample_start": 0,
        "sample_count": 1,
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
    sample = Sample(
        condition=np.zeros((4, 4, 1), dtype=np.float32),
        trajectory=np.zeros((1, 4, 4, 1), dtype=np.float32),
        geometry=np.zeros((4, 4, 1), dtype=np.float32),
        metadata={
            "sample_id": "contract-sample-0",
            "schema_version": "1.0",
            "pde": "poisson",
            "boundary": "periodic",
            "setting": "smooth_grf",
            "regime": "low",
            "state_representation": "scalar",
            "resolution": [4, 4],
            "T": 1,
            "split": "train",
            "seed": 7,
            "parameters": {},
            "solver_fidelity": "compact_reference",
        },
    )
    with AtomicHDF5ShardWriter(path, expected_count=1, spec=spec) as writer:
        writer.append(sample)
    return path


def test_strict_aggregate_rejects_cross_array_spatial_mismatch(tmp_path: Path) -> None:
    path = _write_contract_shard(tmp_path / "spatial.h5")
    with h5py.File(path, "a") as handle:
        del handle["geometry"]
        handle.create_dataset("geometry", data=np.zeros((1, 5, 4, 1), dtype=np.float32))

    with pytest.raises(ShardValidationError, match="geometry spatial dimensions"):
        validate_hdf5_shard(path, verify_checksum=False, strict=True)


def test_strict_aggregate_rejects_cross_array_channel_mismatch(tmp_path: Path) -> None:
    path = _write_contract_shard(tmp_path / "channels.h5")
    with h5py.File(path, "a") as handle:
        del handle["condition"]
        handle.create_dataset("condition", data=np.zeros((1, 4, 4, 2), dtype=np.float32))

    with pytest.raises(ShardValidationError, match="built-in poisson contract"):
        validate_hdf5_shard(path, verify_checksum=False, strict=True)


def test_strict_aggregate_rejects_state_representation_mismatch(tmp_path: Path) -> None:
    path = _write_contract_shard(tmp_path / "representation.h5")
    with h5py.File(path, "a") as handle:
        metadata = json.loads(handle["metadata"][0])
        metadata["state_representation"] = "velocity"
        handle["metadata"][0] = json.dumps(metadata, sort_keys=True, separators=(",", ":"))

    with pytest.raises(ShardValidationError, match="state_representation"):
        validate_hdf5_shard(path, verify_checksum=False, strict=True)


def test_quality_audit_does_not_silently_recompute_missing_records(tmp_path: Path) -> None:
    sample = _generated_sample("poisson")
    valid_metadata = {
        **sample.metadata,
        "sample_id": "valid",
        "quality": evaluate_sample_quality(sample),
    }
    missing_metadata = {**sample.metadata, "sample_id": "missing"}
    invalid_metadata = {**sample.metadata, "sample_id": "invalid", "quality": {}}
    path = tmp_path / "audit.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("condition", data=np.stack([sample.condition] * 3))
        handle.create_dataset("trajectory", data=np.stack([sample.trajectory] * 3))
        handle.create_dataset("geometry", data=np.stack([sample.geometry] * 3))
        handle.create_dataset(
            "metadata",
            data=np.asarray(
                [
                    json.dumps(row, sort_keys=True)
                    for row in (valid_metadata, missing_metadata, invalid_metadata)
                ],
                dtype=object,
            ),
            dtype=string_dtype,
        )

    stored = audit_dataset_quality(tmp_path)
    recomputed = audit_dataset_quality(tmp_path, recompute=True)

    assert stored["sample_count"] == 3
    assert stored["quality"]["input_count"] == 3
    assert stored["quality"]["record_count"] == 1
    assert stored["quality"]["missing_quality_count"] == 1
    assert stored["quality"]["invalid_quality_count"] == 1
    assert stored["gate"]["status"] == "fail"
    assert recomputed["sample_count"] == 3
    assert recomputed["quality"]["record_count"] == 3


def test_new_quality_spec_requires_intact_quality_sidecar(tmp_path: Path) -> None:
    base = _generated_sample("poisson")
    metadata = {
        **base.metadata,
        "sample_id": "quality-required-0",
        "schema_version": "1.0",
        "state_representation": "scalar",
        "resolution": [8, 8],
        "T": 1,
        "split": "train",
        "seed": 31,
        "tier": "tiny",
    }
    provisional = Sample(base.condition, base.trajectory, base.geometry, metadata)
    metadata["quality"] = evaluate_sample_quality(provisional)
    sample = Sample(base.condition, base.trajectory, base.geometry, metadata)
    path = tmp_path / "mandatory-quality.h5"
    spec = {
        "pde": "poisson",
        "boundary": "periodic",
        "setting": "smooth_grf",
        "regime": "low",
        "tier": "tiny",
        "sample_count": 1,
        "quality": {"enabled": True, "profile": "report"},
    }
    with AtomicHDF5ShardWriter(path, expected_count=1, spec=spec) as writer:
        writer.append(sample)

    validate_hdf5_shard(path, verify_checksum=True, strict=True)
    shard_sidecars(path)["quality"].unlink()

    assert is_shard_complete(path) is False
    with pytest.raises(ShardValidationError, match="quality_json"):
        validate_hdf5_shard(path, verify_checksum=True, strict=True)
