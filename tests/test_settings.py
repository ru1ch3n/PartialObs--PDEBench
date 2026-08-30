# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from pdeobs.pdes.common import make_setting_field
from pdeobs.schema import GenerationSpec, Sample
from pdeobs.settings import SETTING_NAMES, generate_setting, list_settings


def test_all_ten_settings_are_deterministic_and_finite():
    assert len(SETTING_NAMES) == 10
    assert set(list_settings()) == set(SETTING_NAMES)
    for name in SETTING_NAMES:
        first = generate_setting(name, (12, 10), seed=41, family="poisson")
        second = generate_setting(name, (12, 10), seed=41, family="poisson")
        assert first.shape == (12, 10)
        assert first.dtype == np.float32
        assert np.array_equal(first, second)
        assert np.all(np.isfinite(first))
        assert np.std(first) > 0


def test_setting_seed_changes_the_field():
    first = generate_setting("smooth_grf", 12, seed=1)
    second = generate_setting("smooth_grf", 12, seed=2)
    assert not np.array_equal(first, second)


def test_pde_generators_use_the_public_setting_registry():
    expected = generate_setting("smooth_grf", 16, seed=23)
    actual = make_setting_field("smooth_grf", 16, np.random.default_rng(23))
    np.testing.assert_allclose(actual, expected)


def test_sample_normalizes_scalar_shape_and_validates_spatial_axes():
    sample = Sample(
        condition=np.zeros((8, 7)),
        trajectory=np.zeros((9, 8, 7)),
        geometry=np.zeros((8, 7)),
        metadata={"seed": np.int64(2)},
    )
    assert sample.condition.shape == (8, 7, 1)
    assert sample.trajectory.shape == (9, 8, 7, 1)
    assert sample.geometry.shape == (8, 7, 1)
    assert sample.metadata["seed"] == 2
    with pytest.raises(ValueError):
        Sample(np.zeros((8, 7)), np.zeros((9, 9, 7)), np.zeros((8, 7)))


def test_generation_spec_round_trip_and_stable_seed():
    spec = GenerationSpec.from_dict(
        {
            "family": "heat",
            "boundary": "periodic",
            "setting": "s0",
            "regime": "low",
            "samples": 5,
            "resolution": [8, 10],
        }
    )
    assert spec.pde == "heat"
    assert spec.spatial_shape == (8, 10)
    assert GenerationSpec.from_dict(spec.to_dict()) == spec
    assert spec.sample_seed(3) == spec.sample_seed(3)
    assert spec.sample_seed(3) != spec.sample_seed(4)


def test_generation_spec_rejects_unsafe_tier():
    with pytest.raises(ValueError, match="tier"):
        GenerationSpec(
            pde="poisson",
            boundary="periodic",
            setting="smooth_grf",
            regime="low",
            tier="../../escape",
        )
