import numpy as np
import pytest

from pdeobs.pdes import (
    BOUNDARY_NAMES,
    PDE_FAMILIES,
    STATIC_FAMILIES,
    generate_sample,
)
from pdeobs.registry import PDE_REGISTRY


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


def test_builtin_pdes_are_available_through_extension_registry():
    assert set(PDE_FAMILIES).issubset(PDE_REGISTRY.names())
    assert PDE_REGISTRY.get("f6") is PDE_REGISTRY.get("navier_stokes")
