# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
import numpy as np

from pdeobs.masks import (
    MASK_PROTOCOL_NAMES,
    apply_mask,
    exact_random_mask,
    generate_mask,
    list_masks,
)


def test_exact_random_mask_has_exact_count_and_is_deterministic():
    first = exact_random_mask((17, 19), count=37, seed=123)
    second = exact_random_mask((17, 19), count=37, seed=123)
    assert first.dtype == np.bool_
    assert first.sum() == 37
    assert np.array_equal(first, second)


def test_main_training_mask_uses_500_points_at_128():
    assert generate_mask("random_3pct", 128, seed=0).sum() == 500


def test_nine_official_protocols_return_valid_deterministic_masks():
    assert len(MASK_PROTOCOL_NAMES) == 9
    assert set(list_masks()) == set(MASK_PROTOCOL_NAMES)
    for protocol in MASK_PROTOCOL_NAMES:
        first = generate_mask(protocol, (24, 20), seed=12)
        second = generate_mask(protocol, (24, 20), seed=12)
        assert first.shape == (24, 20)
        assert first.dtype == np.bool_
        assert first.any()
        assert np.array_equal(first, second)


def test_apply_mask_broadcasts_through_time_and_channels():
    values = np.ones((9, 8, 7, 2), dtype=np.float32)
    mask = exact_random_mask((8, 7), 10, seed=4)
    observed = apply_mask(values, mask, fill_value=-1.0)
    assert np.all(observed[:, mask, :] == 1.0)
    assert np.all(observed[:, ~mask, :] == -1.0)


def test_line_sensor_budget_is_stable_across_seeds():
    counts = {
        int(generate_mask("line_sensors", (128, 128), seed=seed, ratio=0.03).sum())
        for seed in range(200)
    }
    assert counts == {508}
