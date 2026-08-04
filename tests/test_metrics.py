import numpy as np

from pdeobs.metrics import (
    MetricSuite,
    frequency_band_errors,
    high_frequency_energy,
    ood_degradation,
    relative_l2,
    rollout_horizon_metrics,
    stability_metrics,
    vorticity,
)


def test_basic_metrics_identity_and_scale():
    target = np.ones((3, 8, 8))
    assert relative_l2(target, target) == 0.0
    assert np.isclose(relative_l2(2 * target, target), 1.0)
    result = MetricSuite()(target, target)
    assert result["mse"] == 0.0
    assert all(value == 0.0 for value in frequency_band_errors(target, target).values())


def test_high_frequency_energy_distinguishes_checkerboard():
    y, x = np.mgrid[:32, :32]
    smooth = np.ones((32, 32))
    checker = (-1.0) ** (x + y)
    assert high_frequency_energy(checker) > high_frequency_energy(smooth)


def test_rollout_and_stability_metrics():
    target = np.ones((2, 8, 1, 8, 8))
    predicted = target.copy()
    result = rollout_horizon_metrics(predicted, target)
    assert result["rel_l2_h8"] == 0.0
    stable = stability_metrics(predicted)
    assert stable["stability_failure_rate"] == 0.0
    predicted[0, -1] = np.inf
    assert stability_metrics(predicted)["nonfinite_rate"] == 0.5


def test_vorticity_and_ood_degradation():
    y, x = np.mgrid[:16, :16]
    velocity = np.stack((-y, x), axis=-1)
    omega = vorticity(velocity)
    np.testing.assert_allclose(omega, 2.0)
    assert ood_degradation(0.2, 0.5) == 2.5
    assert np.isclose(ood_degradation(0.8, 0.6, higher_is_better=True, mode="difference"), 0.2)
