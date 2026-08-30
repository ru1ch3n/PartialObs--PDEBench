# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_checkpoint_is_weights_only_safe_and_restores_model(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.runner import _load_checkpoint
    from pdeobs.training import Trainer, TrainingConfig, load_checkpoint_payload

    model = torch.nn.Conv2d(1, 1, kernel_size=1)
    trainer = Trainer(
        model,
        TrainingConfig(
            epochs=1,
            amp=False,
            device="cpu",
            checkpoint_dir=str(tmp_path),
        ),
    )
    checkpoint = trainer.save_checkpoint("safe.pt", epoch=1)

    payload = load_checkpoint_payload(checkpoint)
    assert payload["format_version"] == 1
    assert isinstance(payload["rng"]["numpy"], dict)

    restored = torch.nn.Conv2d(1, 1, kernel_size=1)
    _load_checkpoint(restored, checkpoint)
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_checkpoint_loader_rejects_unsafe_pickle_payload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.training import load_checkpoint_payload

    checkpoint = tmp_path / "legacy-unsafe.pt"
    torch.save({"array": np.arange(3)}, checkpoint)

    with pytest.raises(ValueError, match="refuses arbitrary-pickle checkpoints"):
        load_checkpoint_payload(checkpoint)
