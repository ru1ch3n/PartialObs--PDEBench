from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pdeobs.dataset import BenchmarkDataset, collate_benchmark
from pdeobs.schema import Sample
from pdeobs.storage import AtomicHDF5ShardWriter


def test_recovery_dataset_filters_and_applies_exact_mask(tmp_path: Path) -> None:
    shard = tmp_path / "fixture.h5"
    with AtomicHDF5ShardWriter(shard, expected_count=2, spec={"test": True}) as writer:
        for index, split in enumerate(("train", "validation")):
            target = np.full((1, 8, 8, 1), index + 1, dtype=np.float32)
            writer.append(
                Sample(
                    condition=target[0],
                    trajectory=target,
                    geometry=np.zeros((8, 8, 1), dtype=np.float32),
                    metadata={"sample_id": f"sample-{index}", "split": split, "pde": "heat"},
                )
            )

    dataset = BenchmarkDataset(
        shard,
        task="recovery",
        split="train",
        filters={"pde": "heat"},
        mask={"protocol": "random", "count": 5},
        seed=9,
        verify=True,
    )
    row = dataset[0]
    batch = collate_benchmark([row])

    assert len(dataset) == 1
    assert int(row["mask"].sum()) == 5
    assert row["metadata"]["mask_id"] == "random_3pct"
    assert isinstance(row["metadata"]["mask_seed"], int)
    assert row["metadata"]["observation_count"] == 5
    assert batch["observations"].shape == (1, 8, 8, 1)
    assert np.count_nonzero(batch["observations"]) == 5


def test_rollout_uses_requested_sparse_history_and_future_horizon(tmp_path: Path) -> None:
    shard = tmp_path / "temporal.h5"
    trajectory = np.arange(6 * 8 * 8, dtype=np.float32).reshape(6, 8, 8, 1)
    with AtomicHDF5ShardWriter(shard, expected_count=1, spec={"test": True}) as writer:
        writer.append(
            Sample(
                condition=trajectory[0],
                trajectory=trajectory,
                geometry=np.zeros((8, 8, 1), dtype=np.float32),
                metadata={
                    "sample_id": "temporal-0",
                    "split": "train",
                    "pde": "heat",
                    "resolution": [8, 8],
                },
            )
        )

    dataset = BenchmarkDataset(
        shard,
        task="rollout",
        history_steps=2,
        horizon=3,
        mask={"protocol": "random", "count": 7},
    )
    row = dataset[0]

    assert row["observations"].shape == (2, 8, 8, 1)
    assert row["mask"].shape == (2, 8, 8, 1)
    assert int(row["mask"][0].sum()) == 7
    assert row["target"].shape == (3, 8, 8, 1)
    np.testing.assert_array_equal(row["target"], trajectory[2:5])


def test_rollout_rejects_a_short_trajectory_instead_of_truncating(tmp_path: Path) -> None:
    shard = tmp_path / "short-temporal.h5"
    trajectory = np.zeros((4, 8, 8, 1), dtype=np.float32)
    with AtomicHDF5ShardWriter(shard, expected_count=1, spec={"test": True}) as writer:
        writer.append(
            Sample(
                condition=trajectory[0],
                trajectory=trajectory,
                geometry=np.zeros((8, 8, 1), dtype=np.float32),
                metadata={"sample_id": "short-0", "split": "train", "pde": "heat"},
            )
        )

    dataset = BenchmarkDataset(
        shard,
        task="rollout",
        history_steps=2,
        horizon=3,
        mask={"protocol": "random", "count": 7},
    )

    with pytest.raises(ValueError, match=r"2 history \+ 3 future"):
        _ = dataset[0]


def test_forward_masks_the_condition(tmp_path: Path) -> None:
    shard = tmp_path / "forward.h5"
    condition = np.ones((8, 8, 1), dtype=np.float32)
    with AtomicHDF5ShardWriter(shard, expected_count=1, spec={"test": True}) as writer:
        writer.append(
            Sample(
                condition=condition,
                trajectory=condition[None],
                geometry=np.zeros_like(condition),
                metadata={"sample_id": "forward-0", "split": "test", "pde": "poisson"},
            )
        )
    row = BenchmarkDataset(
        shard,
        task="forward",
        mask={"protocol": "random", "count": 5},
    )[0]
    assert int(row["mask"].sum()) == 5
    assert np.count_nonzero(row["observations"]) == 5


def test_inverse_targets_the_original_condition(tmp_path: Path) -> None:
    shard = tmp_path / "inverse.h5"
    condition = np.full((8, 8, 1), 3.0, dtype=np.float32)
    solution = np.full((1, 8, 8, 1), 7.0, dtype=np.float32)
    with AtomicHDF5ShardWriter(shard, expected_count=1, spec={"test": True}) as writer:
        writer.append(
            Sample(
                condition=condition,
                trajectory=solution,
                geometry=np.zeros_like(condition),
                metadata={"sample_id": "inverse-0", "split": "test", "pde": "darcy"},
            )
        )
    row = BenchmarkDataset(
        shard,
        task="inverse",
        mask={"protocol": "random", "count": 6},
    )[0]
    assert int(row["mask"].sum()) == 6
    assert np.count_nonzero(row["observations"]) == 6
    np.testing.assert_array_equal(row["target"], condition)
