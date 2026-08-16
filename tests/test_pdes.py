import numpy as np
import pytest

import pdeobs.pdes as pdes
from pdeobs.pdes import (
    BOUNDARY_NAMES,
    PDE_FAMILIES,
    STATIC_FAMILIES,
    generate_sample,
)
from pdeobs.pdes.common import build_output
from pdeobs.pdes.darcy import CONTRAST_BY_REGIME
from pdeobs.registry import PDE_REGISTRY
from pdeobs.settings import SETTING_NAMES


def test_nonfinite_solver_output_is_rejected_instead_of_sanitized():
    condition = np.zeros((4, 4, 1), dtype=np.float64)
    trajectory = condition[None].copy()
    trajectory[0, 0, 0, 0] = np.nan

    with pytest.raises(FloatingPointError, match="NaN or infinity"):
        build_output(
            family="poisson",
            boundary="periodic",
            setting="smooth_grf",
            regime="low",
            seed=0,
            condition=condition,
            trajectory=trajectory,
            geometry=condition,
            parameters={},
        )


@pytest.mark.parametrize("family", PDE_FAMILIES)
@pytest.mark.parametrize("boundary", BOUNDARY_NAMES)
def test_all_pde_families_and_boundaries_have_canonical_finite_output(family, boundary):
    output = generate_sample(
        family,
        boundary=boundary,
        setting="s0",
        regime="medium",
        seed=7,
        resolution=(8, 10),
    )
    expected_steps = 1 if family in STATIC_FAMILIES else 9
    assert output.condition.shape[:2] == (8, 10)
    assert output.trajectory.shape[:3] == (expected_steps, 8, 10)
    assert output.geometry.shape == (8, 10, 1)
    assert np.all(np.isfinite(output.condition))
    assert np.all(np.isfinite(output.trajectory))
    assert np.all(np.isfinite(output.geometry))


@pytest.mark.parametrize("family", PDE_FAMILIES)
def test_pde_generation_is_seed_deterministic(family):
    kwargs = dict(
        family=family,
        boundary="periodic",
        setting="front_ring_shock",
        regime="high",
        seed=93,
        resolution=8,
    )
    first = generate_sample(**kwargs)
    second = generate_sample(**kwargs)
    assert np.array_equal(first.condition, second.condition)
    assert np.array_equal(first.trajectory, second.trajectory)
    assert np.array_equal(first.geometry, second.geometry)


def test_navier_stokes_state_channels_follow_geometry_protocol():
    periodic = generate_sample("navier_stokes", boundary="periodic", resolution=8)
    obstacle = generate_sample("navier_stokes", boundary="robin_obstacle", resolution=8)
    assert periodic.trajectory.shape == (9, 8, 8, 1)  # vorticity
    assert obstacle.trajectory.shape == (9, 8, 8, 2)  # velocity components
    assert obstacle.geometry[..., 0].any()


@pytest.mark.parametrize("setting", SETTING_NAMES)
@pytest.mark.parametrize("regime", tuple(CONTRAST_BY_REGIME))
def test_darcy_regimes_realize_exact_contrast_across_settings(setting, regime):
    output = generate_sample(
        "darcy",
        boundary="periodic",
        setting=setting,
        regime=regime,
        seed=17,
        resolution=8,
        solver_steps=1,
    )
    coefficient = output.condition[..., 0]
    realized = float(np.max(coefficient) / np.min(coefficient))
    requested = CONTRAST_BY_REGIME[regime]
    assert np.isclose(realized, requested, rtol=2.0e-6)
    assert output.parameters["requested_coefficient_contrast"] == requested
    assert np.isclose(output.parameters["realized_coefficient_contrast"], realized, rtol=1.0e-7)


def test_builtin_pdes_are_available_through_extension_registry():
    assert set(PDE_FAMILIES).issubset(PDE_REGISTRY.names())
    assert PDE_REGISTRY.get("f6") is PDE_REGISTRY.get("navier_stokes")


def test_direct_generation_discovers_an_explicit_builtin_replacement(monkeypatch):
    monkeypatch.setattr(pdes, "FAMILY_GENERATORS", dict(pdes.FAMILY_GENERATORS))
    monkeypatch.setattr(pdes, "_GENERATOR_ALIASES", dict(pdes._GENERATOR_ALIASES))
    monkeypatch.setattr(pdes, "_PLUGINS_DISCOVERED", False)
    monkeypatch.setattr(PDE_REGISTRY, "_objects", dict(PDE_REGISTRY._objects))
    monkeypatch.setattr(PDE_REGISTRY, "_aliases", dict(PDE_REGISTRY._aliases))

    def replacement(**options):
        condition = np.ones((8, 8, 1), dtype=np.float32)
        return pdes.PDEOutput(
            family="poisson",
            boundary=options["boundary"],
            setting=options["setting"],
            regime=options["regime"],
            seed=options["seed"],
            condition=condition,
            trajectory=condition[None],
            geometry=np.zeros_like(condition),
            parameters={"validated_plugin": True},
        )

    def discover(**_options):
        pdes.register_generator("poisson", replacement, replace=True)
        return ("poisson",)

    monkeypatch.setattr(PDE_REGISTRY, "discover", discover)
    output = pdes.generate_sample("poisson", resolution=8)

    assert output.parameters["validated_plugin"] is True
