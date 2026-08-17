import hashlib
import json

import numpy as np
import pytest

from pdeobs.pdes import generate_sample
from pdeobs.quality import (
    BUILTIN_PDE_FAMILIES,
    QUALITY_SCHEMA_VERSION,
    QualityGateError,
    assess_quality_gate,
    enforce_generation_quality,
    evaluate_sample_quality,
    normalize_quality_config,
    summarize_quality_records,
)
from pdeobs.schema import Sample

STATIC_FAMILIES = frozenset({"darcy", "poisson", "helmholtz"})


def _as_sample(output, *, solver_fidelity="compact_reference"):
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
            "solver_fidelity": solver_fidelity,
        },
    )


@pytest.fixture(scope="module")
def periodic_samples():
    samples = {}
    for family in BUILTIN_PDE_FAMILIES:
        output = generate_sample(
            family,
            boundary="periodic",
            setting="smooth_grf",
            regime="low",
            seed=11,
            resolution=12,
            time_steps=None if family in STATIC_FAMILIES else 4,
        )
        samples[family] = _as_sample(output)
    return samples


@pytest.mark.parametrize("family", BUILTIN_PDE_FAMILIES)
def test_every_builtin_family_reports_finite_json_quality(periodic_samples, family):
    report = evaluate_sample_quality(periodic_samples[family])

    assert report["schema_version"] == QUALITY_SCHEMA_VERSION
    assert report["pde"] == family
    assert report["profile"] == "report"
    assert report["pde_loss"]["available"] is True
    assert report["operator"]
    assert report["metrics"]["finite_fraction"] == 1.0
    assert report["metrics"]["pde_loss_normalized"] is not None
    assert np.isfinite(report["metrics"]["pde_loss_normalized"])
    assert report["checks"]["pde_loss"] == "reported"
    # A report-only residual deliberately cannot become a silent acceptance gate.
    assert report["status"] == "warning"
    assert report["publication_ready"] is False
    json.dumps(report, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize(
    "family",
    ("heat", "reaction_diffusion", "burgers", "navier_stokes"),
)
def test_periodic_temporal_quality_uses_declared_spectral_operator(periodic_samples, family):
    report = evaluate_sample_quality(periodic_samples[family])

    assert "spectral" in report["operator_id"]
    assert (
        report["operator_id"]
        == periodic_samples[family].metadata["parameters"]["quality_residual_contract"]
    )


@pytest.mark.parametrize(
    "family",
    ("heat", "reaction_diffusion", "burgers", "navier_stokes"),
)
def test_temporal_quality_separates_initial_replay_from_saved_frame_balance(
    periodic_samples, family
):
    report = evaluate_sample_quality(periodic_samples[family])

    assert report["checks"]["initial_transition_contract"] == "pass"
    assert report["checks"]["initial_transition_replay"] == "pass"
    assert report["metrics"]["initial_transition_replay_loss_normalized"] < 5.0e-6
    assert report["metrics"]["pde_loss_all_steps_normalized"] is not None
    assert report["metrics"]["pde_loss_first_step_strong_normalized"] is not None
    assert report["pde_loss"]["normalized"] == report["pde_loss"]["post_initial_normalized"]


def test_periodic_heat_uses_exact_semigroup_loss_and_reports_fd2_strong_form():
    output = generate_sample(
        "heat",
        boundary="periodic",
        setting="multi_frequency_fourier",
        regime="high",
        seed=17,
        resolution=32,
        time_steps=33,
    )
    report = evaluate_sample_quality(
        _as_sample(output),
        config={"profile": "strict", "thresholds": {"pde_loss_normalized_max": 5.0e-5}},
    )

    assert report["status"] == "pass"
    assert report["operator"] == "u[n+1]=exp(D*dt*laplace)u[n]"
    assert "spectral_semigroup" in report["operator_id"]
    assert report["metrics"]["pde_loss_normalized"] < 5.0e-5
    assert report["metrics"]["initial_transition_replay_loss_normalized"] < 5.0e-6
    assert report["metrics"]["auxiliary_fd2_strong_form_normalized"] is not None
    assert (
        report["metrics"]["pde_loss_first_step_strong_normalized"]
        == report["metrics"]["auxiliary_fd2_strong_form_first_step_normalized"]
    )


@pytest.mark.parametrize("family", BUILTIN_PDE_FAMILIES)
def test_family_residual_is_sensitive_to_grid_scale_trajectory_error(periodic_samples, family):
    sample = periodic_samples[family]
    baseline = evaluate_sample_quality(sample)["metrics"]["pde_loss_normalized"]

    corrupted = sample.trajectory.astype(np.float64).copy()
    yy, xx = np.indices(sample.spatial_shape)
    checkerboard = (2 * ((xx + yy) % 2) - 1).astype(np.float64)
    amplitude = 0.5 * max(float(np.sqrt(np.mean(corrupted**2))), 1.0e-2)
    for frame in range(corrupted.shape[0]):
        # Preserve the stored initial state for temporal families so this test
        # changes the PDE defect without relying on the initial-condition check.
        if corrupted.shape[0] == 1 or frame > 0:
            corrupted[frame] += ((-1.0) ** frame) * amplitude * checkerboard[..., None]
    perturbed = Sample(sample.condition, corrupted, sample.geometry, sample.metadata)
    observed = evaluate_sample_quality(perturbed)["metrics"]["pde_loss_normalized"]

    assert observed > baseline


def test_quality_profiles_separate_reporting_enforcement_and_publication(periodic_samples):
    sample = periodic_samples["poisson"]

    report = evaluate_sample_quality(sample)
    strict = evaluate_sample_quality(
        sample,
        config={"profile": "strict", "thresholds": {"pde_loss_normalized_max": 1.0}},
    )
    rejected = evaluate_sample_quality(
        sample,
        config={"profile": "strict", "thresholds": {"pde_loss_normalized_max": 0.0}},
    )
    publication = evaluate_sample_quality(
        sample,
        config={"profile": "publication", "thresholds": {"pde_loss_normalized_max": 1.0}},
    )

    assert report["status"] == "warning"
    assert strict["status"] == "pass"
    enforce_generation_quality(strict)
    assert rejected["status"] == "fail"
    with pytest.raises(QualityGateError, match="pde_loss"):
        enforce_generation_quality(rejected)
    assert publication["status"] == "fail"
    assert publication["checks"]["pde_threshold_frozen"] == "fail"
    assert publication["checks"]["validated_solver"] == "fail"
    assert publication["publication_ready"] is False


def test_publication_profile_requires_an_explicit_threshold_and_validated_solver(
    periodic_samples,
):
    compact = periodic_samples["poisson"]
    implementation = "independent.poisson:ValidatedSolver"
    version = "1.2.3"
    evidence = {
        "schema_version": "pdeobs.numerical-validation/v1",
        "report_sha256": "a" * 64,
        "solver_artifact_sha256": "b" * 64,
        "solver_implementation": implementation,
        "solver_version": version,
    }
    validated = Sample(
        compact.condition,
        compact.trajectory,
        compact.geometry,
        {
            **compact.metadata,
            "solver_fidelity": "validated_reference",
            "solver_implementation": implementation,
            "solver_version": version,
            "solver_validation_evidence": evidence,
        },
    )
    calibration_key = evaluate_sample_quality(validated)["calibration_key"]
    threshold_table = {calibration_key: 1.0}
    threshold_digest = hashlib.sha256(
        json.dumps(threshold_table, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    unfrozen = evaluate_sample_quality(validated, config={"profile": "publication"})
    accepted = evaluate_sample_quality(
        validated,
        config={
            "profile": "publication",
            "thresholds": {"pde_loss_normalized_max": 1.0},
            "calibration_evidence": {
                "schema_version": "pdeobs.quality-thresholds/v1",
                "table_sha256": threshold_digest,
                "covered_keys": [calibration_key],
                "pde_loss_normalized_max_by_key": threshold_table,
            },
        },
    )

    assert unfrozen["status"] == "fail"
    assert unfrozen["checks"]["pde_threshold_frozen"] == "fail"
    assert accepted["status"] == "fail"
    assert accepted["checks"]["solver_validation_evidence"] == "pass"
    assert accepted["checks"]["independent_evidence_verification"] == "fail"
    assert accepted["sample_quality_attestation_complete"] is True
    assert accepted["sample_quality_gate_ready"] is False
    assert accepted["publication_ready"] is False


def test_publication_profile_selects_effective_threshold_by_calibration_key(
    periodic_samples,
):
    compact = periodic_samples["poisson"]
    implementation = "independent.poisson:ValidatedSolver"
    version = "1.2.3"
    evidence = {
        "schema_version": "pdeobs.numerical-validation/v1",
        "report_sha256": "a" * 64,
        "solver_artifact_sha256": "b" * 64,
        "solver_implementation": implementation,
        "solver_version": version,
    }
    common_metadata = {
        **compact.metadata,
        "solver_fidelity": "validated_reference",
        "solver_implementation": implementation,
        "solver_version": version,
        "solver_validation_evidence": evidence,
    }
    first = Sample(compact.condition, compact.trajectory, compact.geometry, common_metadata)
    second = Sample(
        compact.condition,
        compact.trajectory,
        compact.geometry,
        {
            **common_metadata,
            "parameters": {
                **common_metadata["parameters"],
                "solver_steps": int(common_metadata["parameters"]["solver_steps"]) + 1,
            },
        },
    )
    first_key = evaluate_sample_quality(first)["calibration_key"]
    second_key = evaluate_sample_quality(second)["calibration_key"]
    threshold_table = {first_key: 10.0, second_key: 20.0}
    threshold_digest = hashlib.sha256(
        json.dumps(threshold_table, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    config = {
        "profile": "publication",
        "thresholds": {"pde_loss_normalized_max": 999.0},
        "calibration_evidence": {
            "schema_version": "pdeobs.quality-thresholds/v1",
            "table_sha256": threshold_digest,
            "covered_keys": [first_key, second_key],
            "pde_loss_normalized_max_by_key": threshold_table,
        },
    }

    first_report = evaluate_sample_quality(first, config=config)
    second_report = evaluate_sample_quality(second, config=config)

    assert first_key != second_key
    assert first_report["thresholds"]["pde_loss_normalized_max"] == 10.0
    assert second_report["thresholds"]["pde_loss_normalized_max"] == 20.0
    assert first_report["checks"]["calibrated_threshold_evidence"] == "pass"
    assert second_report["checks"]["calibrated_threshold_evidence"] == "pass"


def test_all_solid_geometry_fails_active_domain_and_cannot_pass_publication(
    periodic_samples,
):
    sample = periodic_samples["poisson"]
    all_solid = Sample(
        sample.condition,
        sample.trajectory,
        np.ones_like(sample.geometry),
        sample.metadata,
    )

    report = evaluate_sample_quality(all_solid)

    assert report["active_spatial_cells"] == 0
    assert report["checks"]["active_domain"] == "fail"
    assert report["pde_loss"]["available"] is False
    assert report["status"] == "fail"
    assert report["publication_ready"] is False


def test_helmholtz_report_is_nominal_equation_loss_not_a_validation_claim(periodic_samples):
    sample = periodic_samples["helmholtz"]
    assert sample.metadata["parameters"]["solver_id"] == "fd2_helmholtz_krylov_v2"

    report = evaluate_sample_quality(sample)

    assert report["operator"] == "(-laplace-k^2)u=f"
    assert report["pde_loss"]["interpretation"].startswith("nominal_equation_residual")
    assert "helmholtz_transfer_loss_normalized" not in report["metrics"]
    assert report["pde_loss"]["status"] == "measured"
    assert report["checks"]["pde_loss"] == "reported"
    assert report["status"] == "warning"
    assert report["publication_ready"] is False


def test_bounded_navier_stokes_reports_vorticity_and_divergence():
    output = generate_sample(
        "navier_stokes",
        boundary="dirichlet",
        setting="smooth_grf",
        regime="low",
        seed=17,
        resolution=12,
        time_steps=4,
    )
    sample = _as_sample(output)
    report = evaluate_sample_quality(sample)

    assert sample.trajectory.shape[-1] == 1
    assert report["operator"] == "omega_t+u*omega_x+v*omega_y=nu*laplace(omega)"
    assert report["pde_loss"]["available"] is True
    assert report["pde_loss"]["status"] == "measured"
    assert np.isfinite(report["metrics"]["divergence_loss_normalized"])
    assert report["checks"]["incompressibility"] == "reported"
    assert report["status"] == "warning"
    assert report["publication_ready"] is False


def test_summary_keeps_explicit_rows_for_all_pde_losses(periodic_samples):
    rows = [
        {
            "pde": family,
            "quality": evaluate_sample_quality(sample),
        }
        for family, sample in periodic_samples.items()
    ]
    summary = summarize_quality_records(rows)

    assert summary["record_count"] == len(BUILTIN_PDE_FAMILIES)
    assert summary["complete_pde_coverage"] is True
    assert summary["missing_pdes"] == []
    assert set(summary["pde_losses"]) == set(BUILTIN_PDE_FAMILIES)
    for family in BUILTIN_PDE_FAMILIES:
        family_loss = summary["pde_losses"][family]
        assert family_loss["status"] == "present"
        assert family_loss["sample_count"] == 1
        assert family_loss["pde_loss_normalized"]["count"] == 1
    json.dumps(summary, allow_nan=False, sort_keys=True)


def test_dataset_gate_rejects_missing_records_and_partial_pde_losses(periodic_samples):
    rows = [
        {"sample_id": family, "quality": evaluate_sample_quality(sample)}
        for family, sample in periodic_samples.items()
    ]
    rows.append({"sample_id": "missing-quality", "pde": "poisson"})
    bounded = generate_sample(
        "navier_stokes",
        boundary="dirichlet",
        setting="smooth_grf",
        regime="low",
        seed=19,
        resolution=12,
        time_steps=4,
    )
    rows[-2] = {
        "sample_id": "bounded-ns",
        "quality": evaluate_sample_quality(
            _as_sample(bounded, solver_fidelity="validated_reference")
        ),
    }
    summary = summarize_quality_records(rows)
    gate = assess_quality_gate(
        summary,
        strict=True,
        max_pde_loss=1.0,
        require_all_pdes=True,
        require_validated_solvers=True,
        expected_record_count=len(rows),
    )

    assert summary["missing_quality_count"] == 1
    assert gate["status"] == "fail"
    assert any("no quality record" in reason for reason in gate["reasons"])
    assert summary["by_pde"]["navier_stokes"]["pde_loss_status_counts"].get("partial", 0) == 0
    assert gate["publication_ready"] is False


@pytest.mark.parametrize("profile", ("report", "strict", "publication"))
def test_normalized_quality_config_has_stable_versioned_profiles(profile):
    config = normalize_quality_config({"profile": profile})

    assert config["schema_version"] == QUALITY_SCHEMA_VERSION
    assert config["profile"] == profile
    assert config["require_pde_loss"] is True
    assert config["thresholds"]["pde_loss_normalized_max"] is None


def test_quality_reporting_cannot_be_disabled():
    with pytest.raises(ValueError, match="cannot be disabled"):
        normalize_quality_config({"enabled": False})
