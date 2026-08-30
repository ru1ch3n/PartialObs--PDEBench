# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Balanced physical regimes, IID splits, OOD labels, and nested release tiers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .schema import derive_seed

REGIMES = ("low", "medium", "high")
SPLITS = ("train", "validation", "test")
SPLIT_WEIGHTS = (0.70, 0.15, 0.15)
TIER_SIZES = {
    "tiny": 5,
    "debug": 20,
    "signal": 100,
    "medium": 500,
    "full": 2000,
}


def allocate_counts(
    total: int, names: Sequence[str], weights: Sequence[float] | None = None
) -> dict[str, int]:
    """Allocate an integer total by deterministic largest remainder."""

    if total < 0:
        raise ValueError("total must be non-negative")
    if not names:
        raise ValueError("at least one name is required")
    if len(set(names)) != len(names):
        raise ValueError("allocation names must be unique")
    selected_weights = (
        np.ones(len(names), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if (
        selected_weights.shape != (len(names),)
        or np.any(selected_weights < 0)
        or selected_weights.sum() <= 0
    ):
        raise ValueError("weights must be non-negative and match names")
    quotas = total * selected_weights / selected_weights.sum()
    counts = np.floor(quotas).astype(int)
    remainder = int(total - counts.sum())
    # Stable sorting makes ties go to the earlier semantic category.  Thus the
    # full 2000 allocation is exactly low=667, medium=667, high=666.
    order = sorted(range(len(names)), key=lambda index: (-(quotas[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return {name: int(count) for name, count in zip(names, counts, strict=True)}


def regime_counts(total: int = 2000) -> dict[str, int]:
    return allocate_counts(total, REGIMES)


def split_counts(total: int = 2000) -> dict[str, int]:
    return allocate_counts(total, SPLITS, SPLIT_WEIGHTS)


def resolve_tier(tier: str | int, *, full_size: int = 2000) -> int:
    if isinstance(tier, (int, np.integer)):
        count = int(tier)
    else:
        normalized = str(tier).strip().lower()
        if normalized not in TIER_SIZES:
            raise ValueError(f"unknown tier {tier!r}; choose from {tuple(TIER_SIZES)}")
        count = TIER_SIZES[normalized]
    if not 1 <= count <= full_size:
        raise ValueError(f"tier size must be between 1 and {full_size}")
    return count


def tier_regime_counts(tier: str | int, *, full_size: int = 2000) -> dict[str, int]:
    """Resolve a tier across regimes while retaining exact balance and nesting."""

    return regime_counts(resolve_tier(tier, full_size=full_size))


def nested_tier_indices(
    tier: str | int,
    *,
    total: int = 2000,
    seed: int = 0,
    shuffle: bool = False,
) -> np.ndarray:
    """Return a prefix of one stable order; every smaller tier is a subset."""

    size = resolve_tier(tier, full_size=total)
    indices = np.arange(total, dtype=np.int64)
    if shuffle:
        np.random.default_rng(int(seed)).shuffle(indices)
    return indices[:size]


def contiguous_label(index: int, counts: Mapping[str, int]) -> str:
    if index < 0 or index >= sum(counts.values()):
        raise IndexError(index)
    offset = 0
    for name, count in counts.items():
        if index < offset + count:
            return name
        offset += count
    raise AssertionError("unreachable")


def assign_regime(sample_index: int, *, total: int = 2000) -> str:
    return contiguous_label(sample_index, regime_counts(total))


def assign_iid_split(sample_index: int, *, total: int = 2000) -> str:
    return contiguous_label(sample_index, split_counts(total))


@dataclass(frozen=True, slots=True)
class SampleAssignment:
    sample_index: int
    regime: str
    split: str
    tier_rank: int

    def in_tier(self, tier: str | int, *, total: int = 2000) -> bool:
        return self.tier_rank < resolve_tier(tier, full_size=total)


def _labels_from_counts(counts: Mapping[str, int]) -> np.ndarray:
    return np.concatenate([np.repeat(name, count) for name, count in counts.items()]).astype(object)


def build_split_plan(
    total: int = 2000,
    *,
    seed: int = 0,
    case_key: str = "",
    shuffle: bool = True,
) -> tuple[SampleAssignment, ...]:
    """Build an exact, deterministic plan for one 2000-sample macro case.

    Regimes use their canonical contiguous ranges and tier ranks interleave
    those ranges in round-robin order.  Consequently ``assignment.regime`` is
    the same regime addressed by generation's ``(regime, regime_index)`` pair,
    while ``assignment.in_tier(...)`` exactly matches the balanced prefix used
    by every nested release tier.  IID splits use an independent derived RNG
    stream and retain exact macro-case counts.
    """

    if total < 1:
        raise ValueError("total must be positive")
    regimes = _labels_from_counts(regime_counts(total))
    splits = _labels_from_counts(split_counts(total))
    if shuffle:
        np.random.default_rng(derive_seed(seed, case_key, "split")).shuffle(splits)

    # Release tiers allocate their requested size across REGIMES by largest
    # remainder, then take a prefix within each regime.  The equivalent global
    # order is low[0], medium[0], high[0], low[1], ... .  Encoding that order in
    # tier_rank makes SampleAssignment authoritative instead of retaining a
    # second, contradictory random tier assignment that generation ignored.
    seen = {regime: 0 for regime in REGIMES}
    regime_order = {regime: index for index, regime in enumerate(REGIMES)}
    rank = np.empty(total, dtype=np.int64)
    for index, regime_value in enumerate(regimes):
        regime = str(regime_value)
        rank[index] = len(REGIMES) * seen[regime] + regime_order[regime]
        seen[regime] += 1
    if sorted(int(value) for value in rank) != list(range(total)):
        raise AssertionError("balanced regime order did not produce exact tier ranks")
    return tuple(
        SampleAssignment(index, str(regimes[index]), str(splits[index]), int(rank[index]))
        for index in range(total)
    )


def official_ood_labels(
    *,
    pde: str | None = None,
    boundary: str,
    setting: str,
    regime: str,
    held_out_boundary: str = "robin_obstacle",
    held_out_settings: Iterable[str] = ("dipole_vortex_pair", "front_ring_shock"),
    held_out_regime: str = "high",
    held_out_combinations: Iterable[tuple[str, str, str]] | None = None,
) -> dict[str, bool]:
    """Return reusable flags for the official factorized OOD protocols."""

    setting_holdout = set(held_out_settings)
    boundary_ood = boundary == held_out_boundary
    setting_ood = setting in setting_holdout
    parameter_ood = regime == held_out_regime
    combinations = set(
        held_out_combinations or (("navier_stokes", "robin_obstacle", "dipole_vortex_pair"),)
    )
    return {
        "boundary_ood": boundary_ood,
        "setting_ood": setting_ood,
        "parameter_ood": parameter_ood,
        "combination_ood": pde is not None and (pde, boundary, setting) in combinations,
    }


def boundary_ood_split(boundary: str, *, held_out: str = "robin_obstacle") -> str:
    """Train on three protocols and test on the held-out fourth protocol."""

    return "test" if boundary == held_out else "train"


def setting_ood_split(
    setting: str,
    *,
    held_out: Iterable[str] = ("dipole_vortex_pair", "front_ring_shock"),
) -> str:
    return "test" if setting in set(held_out) else "train"


def parameter_ood_split(regime: str, *, held_out: str = "high") -> str:
    return "test" if regime == held_out else "train"


def combination_ood_split(
    pde: str,
    boundary: str,
    setting: str,
    *,
    held_out: Iterable[tuple[str, str, str]] | None = None,
) -> str:
    """Hold out combinations while allowing every individual factor in train."""

    combinations = set(held_out or (("navier_stokes", "robin_obstacle", "dipole_vortex_pair"),))
    return "test" if (pde, boundary, setting) in combinations else "train"


def mask_ood_split(protocol: str, *, training_protocol: str = "random_3pct") -> str:
    return "train" if protocol == training_protocol else "test"


def time_horizon_ood_split(horizon: int, *, training_horizons: Iterable[int] = (1, 2)) -> str:
    if int(horizon) < 1:
        raise ValueError("horizon must be positive")
    return "train" if int(horizon) in set(training_horizons) else "test"
