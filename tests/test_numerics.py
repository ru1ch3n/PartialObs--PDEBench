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
    assert report["pde_loss"]["available"] is True
    assert np.isfinite(report["metrics"]["pde_loss_normalized"])


@pytest.mark.parametrize("boundary", ("dirichlet", "neumann", "robin"))
def test_bounded_navier_stokes_uses_pressure_projection(boundary: str) -> None:
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

    assert output.parameters["integrator_id"] == "mac_projection_fd2_v2"
    assert output.trajectory.shape[-1] == 2
    assert output.parameters["pressure_relative_residual_max"] < 1.0e-6
    assert report["pde_loss"]["available"] is True
    assert report["pde_loss"]["status"] == "partial"
