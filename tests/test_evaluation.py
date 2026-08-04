from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from pdeobs.evaluation import EvaluationConfig, evaluate_model, evaluate_predictions


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

    result = evaluate_model(
        IdentityMethod(),
        [_batch(target[:2], target[:2]), _batch(target[2:], target[2:])],
        config=config,
    )

    assert result["samples"] == 3
    assert output.is_file()
    with h5py.File(output, "r") as handle:
        assert handle["prediction"].shape == (3, 8, 8, 1)
        assert handle["target"].shape == (3, 8, 8, 1)
        assert int(handle.attrs["samples"]) == 3
