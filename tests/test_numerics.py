from __future__ import annotations

import numpy as np
import pytest

from pdeobs.pdes import generate_sample
from pdeobs.quality import evaluate_sample_quality
from pdeobs.schema import Sample


def _sample(output) -> Sample:
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
            "solver_fidelity": "numerics_validation_candidate",
        },
    )


@pytest.mark.parametrize("family", ("darcy", "poisson", "helmholtz"))
@pytest.mark.parametrize("boundary", ("dirichlet", "neumann", "periodic", "robin"))
def test_static_solvers_converge_with_boundary_in_operator(family: str, boundary: str) -> None:
    output = generate_sample(
        family,
        boundary=boundary,
        setting="smooth_grf",
        regime="medium",
        seed=31,
        resolution=20,
    )
    report = evaluate_sample_quality(_sample(output))

    assert output.parameters["solver_relative_residual"] < 1.0e-7
    assert report["pde_loss"]["available"] is True
    assert np.isfinite(report["metrics"]["pde_loss_normalized"])
    assert report["metrics"]["pde_loss_normalized"] < 2.0e-3


@pytest.mark.parametrize("family", ("heat", "reaction_diffusion", "burgers"))
@pytest.mark.parametrize("boundary", ("dirichlet", "neumann", "periodic", "robin"))
def test_scalar_temporal_solver_never_uses_periodic_step_for_bounded_bc(
    family: str, boundary: str
) -> None:
    output = generate_sample(
        family,
        boundary=boundary,
        setting="smooth_grf",
        regime="low",
        seed=32,
        resolution=16,
        time_steps=3,
    )
    integrator = str(output.parameters["integrator_id"])
    report = evaluate_sample_quality(_sample(output))

    assert np.all(np.isfinite(output.trajectory))
    assert report["pde_loss"]["available"] is True
    if boundary == "periodic":
        assert "fourier" in integrator or "pseudospectral" in integrator
    else:
        assert "boundary_v2" in integrator
    if family == "burgers":
        assert output.parameters["max_frame_courant"] <= 0.35 * (1.0 + 1.0e-12)


def test_periodic_navier_stokes_uses_fno_protocol_and_reports_forcing() -> None:
    output = generate_sample(
        "navier_stokes",
        boundary="periodic",
        setting="smooth_grf",
        regime="low",
        seed=33,
        resolution=16,
        time_steps=3,
    )
    report = evaluate_sample_quality(_sample(output))

    assert output.parameters["integrator_id"] == "fno_dealiased_vorticity_cn_v2"
    assert output.parameters["forcing_id"] == "fno_sine_cosine_v1"
    assert output.parameters["solver_route"] == "fno_spectral_vorticity"
    assert output.parameters["internal_time_step"] == pytest.approx(1.0e-4)
    assert report["pde_loss"]["available"] is True
    assert np.isfinite(report["metrics"]["pde_loss_normalized"])


@pytest.mark.parametrize("setting", ("piecewise_blocks", "threshold_level_set", "front_ring_shock"))
def test_periodic_nonsmooth_burgers_uses_conservative_finite_volume_route(
    setting: str,
) -> None:
    output = generate_sample(
        "burgers",
        boundary="periodic",
        setting=setting,
        regime="low",
        seed=35,
        resolution=16,
        time_steps=5,
    )
    report = evaluate_sample_quality(_sample(output))

    assert output.parameters["advection_scheme"] == "rusanov"
    assert output.parameters["integrator_id"] == ("periodic_rusanov_fv_fourier_diffusion_v2")
    assert "rusanov" in report["operator_id"]
    assert report["pde_loss"]["available"] is True


@pytest.mark.parametrize("boundary", ("dirichlet", "neumann", "robin"))
def test_bounded_navier_stokes_uses_topology_matched_vorticity_solver(
    boundary: str,
) -> None:
    output = generate_sample(
        "navier_stokes",
        boundary=boundary,
        setting="smooth_grf",
        regime="low",
        seed=34,
        resolution=12,
        time_steps=2,
    )
    report = evaluate_sample_quality(_sample(output))

    if boundary in {"dirichlet", "neumann"}:
        assert output.parameters["integrator_id"] == "dst_vorticity_streamfunction_ssprk2_v1"
        assert output.trajectory.shape[-1] == 1
        assert output.parameters["state_representation"] == "bounded_vorticity"
    else:
        assert output.parameters["integrator_id"] == "masked_vorticity_streamfunction_ssprk2_v1"
        assert output.trajectory.shape[-1] == 1
        assert output.parameters["state_representation"] == "bounded_obstacle_vorticity"
    assert output.parameters["pressure_relative_residual_max"] < 1.0e-6
    assert report["pde_loss"]["available"] is True
    assert report["pde_loss"]["status"] == "measured"
    assert report["metrics"]["divergence_loss_normalized"] < 1.0e-6
    assert report["checks"]["boundary_condition"] == "pass"
