# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
from collections import Counter

from pdeobs.splits import (
    TIER_SIZES,
    assign_regime,
    boundary_ood_split,
    build_split_plan,
    combination_ood_split,
    mask_ood_split,
    nested_tier_indices,
    official_ood_labels,
    parameter_ood_split,
    regime_counts,
    setting_ood_split,
    split_counts,
    tier_regime_counts,
    time_horizon_ood_split,
)


def test_official_counts_resolve_integer_remainders_exactly():
    assert regime_counts(2000) == {"low": 667, "medium": 667, "high": 666}
    assert split_counts(2000) == {"train": 1400, "validation": 300, "test": 300}
    assert sum(tier_regime_counts("tiny").values()) == 5
    assert tier_regime_counts("tiny") == {"low": 2, "medium": 2, "high": 1}


def test_release_tiers_are_nested_prefixes():
    previous = set()
    for tier, expected_size in TIER_SIZES.items():
        indices = nested_tier_indices(tier, seed=77, shuffle=True)
        assert len(indices) == expected_size
        assert previous.issubset(set(indices))
        previous = set(indices)


def test_split_plan_is_deterministic_and_exact():
    first = build_split_plan(seed=8, case_key="poisson/periodic/s0")
    second = build_split_plan(seed=8, case_key="poisson/periodic/s0")
    assert first == second
    assert Counter(row.regime for row in first) == regime_counts()
    assert Counter(row.split for row in first) == split_counts()
    assert sorted(row.tier_rank for row in first) == list(range(2000))
    assert all(row.regime == assign_regime(row.sample_index) for row in first)

    previous: set[int] = set()
    for tier, expected_size in TIER_SIZES.items():
        selected = {row.sample_index for row in first if row.in_tier(tier)}
        assert len(selected) == expected_size
        assert previous.issubset(selected)
        previous = selected


def test_factorized_ood_split_helpers():
    assert boundary_ood_split("robin_obstacle") == "test"
    assert setting_ood_split("front_ring_shock") == "test"
    assert parameter_ood_split("high") == "test"
    assert combination_ood_split("navier_stokes", "robin_obstacle", "dipole_vortex_pair") == "test"
    assert mask_ood_split("random_3pct") == "train"
    assert mask_ood_split("regular_grid") == "test"
    assert time_horizon_ood_split(2) == "train"
    assert time_horizon_ood_split(8) == "test"
    labels = official_ood_labels(
        pde="navier_stokes",
        boundary="robin_obstacle",
        setting="dipole_vortex_pair",
        regime="high",
    )
    assert labels == {
        "boundary_ood": True,
        "setting_ood": True,
        "parameter_ood": True,
        "combination_ood": True,
    }
