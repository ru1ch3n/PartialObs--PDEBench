# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from pdeobs.difficulty import analyze_path
from pdeobs.evaluation import (
    EvaluationConfig,
    evaluate_model,
    evaluate_prediction_file,
    evaluate_predictions,
)
from pdeobs.reports import load_records


class IdentityMethod:
    name = "identity"

    def predict(self, observations, mask=None, **kwargs):
        del mask, kwargs
        return np.asarray(observations)


def _batch(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    return {"input": prediction, "target": target}


def test_physical_metrics_default_to_channels_last() -> None:
    target = np.zeros((1, 8, 8, 2), dtype=np.float32)
    target[..., 1] = np.linspace(0.0, 1.0, 8)[None, None, :]
    prediction = target.copy()
    prediction[..., 1] *= 0.5

    metrics = evaluate_predictions(
        prediction,
        target,
        EvaluationConfig(physical_representation="velocity"),
    )

    assert metrics["vorticity_rel_l2"] > 0.0
    assert metrics["energy_relative_error"] > 0.0


def test_physical_metrics_exclude_solid_geometry() -> None:
    target = np.ones((1, 8, 8, 2), dtype=np.float32)
    prediction = target.copy()
    prediction[:, 2:6, 2:6] = 100.0
    geometry = np.zeros((1, 8, 8, 1), dtype=np.float32)
    geometry[:, 2:6, 2:6] = 1.0
    metrics = evaluate_predictions(
        prediction,
        target,
        EvaluationConfig(physical_representation="velocity"),
        geometry=geometry,
    )
    assert metrics["energy_relative_error"] == 0.0


def test_streamed_metrics_do_not_depend_on_batch_boundaries() -> None:
    target = np.stack(
        [np.ones((8, 8, 1), dtype=np.float32), np.full((8, 8, 1), 100.0, dtype=np.float32)]
    )
    prediction = np.stack((np.zeros_like(target[0]), target[1] * 0.9))
    config = EvaluationConfig(include_stability=False)

    together = evaluate_model(IdentityMethod(), [_batch(prediction, target)], config=config)
    separate = evaluate_model(
        IdentityMethod(),
        [_batch(prediction[:1], target[:1]), _batch(prediction[1:], target[1:])],
        config=config,
    )

    assert together["metrics"] == separate["metrics"]


def test_evaluate_model_streams_prediction_export(tmp_path: Path) -> None:
    target = np.ones((3, 8, 8, 1), dtype=np.float32)
    output = tmp_path / "predictions.h5"
    config = EvaluationConfig(predictions_path=str(output), include_stability=False)
    mask = np.ones_like(target)
    geometry = np.zeros_like(target)
    metadata = [{"sample_id": f"sample-{index}", "pde": "poisson"} for index in range(3)]

    result = evaluate_model(
        IdentityMethod(),
        [
            {
                "input": target[:2],
                "mask": mask[:2],
                "target": target[:2],
                "geometry": geometry[:2],
                "metadata": metadata[:2],
            },
            {
                "input": target[2:],
                "mask": mask[2:],
                "target": target[2:],
                "geometry": geometry[2:],
                "metadata": metadata[2:],
            },
        ],
        config=config,
    )

    assert result["samples"] == 3
    assert output.is_file()
    with h5py.File(output, "r") as handle:
        assert handle["prediction"].shape == (3, 8, 8, 1)
        assert handle["target"].shape == (3, 8, 8, 1)
        assert handle["observation"].shape == (3, 8, 8, 1)
        assert handle["mask"].shape == (3, 8, 8, 1)
        assert handle["geometry"].shape == (3, 8, 8, 1)
        assert handle["sample_id"].asstr()[0] == "sample-0"
        assert '"pde": "poisson"' in handle["metadata_json"].asstr()[0]
        assert int(handle.attrs["samples"]) == 3


def test_prediction_file_eval_filters_groups_and_reports_unavailable_physics(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.h5"
    target = np.ones((3, 8, 8, 1), dtype=np.float32)
    with h5py.File(prediction_path, "w") as handle:
        handle.create_dataset("prediction", data=target * 0.5)
        handle.create_dataset("target", data=target)

    report = evaluate_prediction_file(
        prediction_path,
        task="sparse_recovery",
        metrics=("rel_l2", "spectral", "pde_residual"),
    )

    assert report["task"] == "recovery"
    assert report["samples"] == 3
    assert report["metrics"]["relative_l2"] == 0.5
    assert "spectral_low" in report["metrics"]
    assert "pde_residual" in report["unavailable_metrics"]
    assert Path(report["output"]).is_file()
    records = load_records(report["sample_records"])
    assert len(records) == 3
    assert records[0]["metrics.relative_l2"] == 0.5
    difficulty = analyze_path(report["sample_records"], tmp_path / "difficulty.json")
    assert difficulty["record_count"] == 3
    assert difficulty["detected"]["primary_metric"] == "relative_l2"
