# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from pdeobs.methods import (
    MethodCapabilities,
    available_methods,
    create_method,
    create_model,
    register_method,
)
from pdeobs.methods import base as method_base
from pdeobs.registry import METHOD_REGISTRY as PROJECT_METHOD_REGISTRY


def sparse_plane():
    y, x = np.mgrid[:9, :9]
    field = (2 * y + x)[..., None].astype(float)
    mask = np.zeros_like(field, dtype=bool)
    mask[::4, ::4] = True
    return np.where(mask, field, 0.0), mask, field


def test_fill_baselines_keep_observations():
    observed, mask, _ = sparse_plane()
    for name in ("zero", "mean", "nearest", "bilinear", "rbf"):
        predicted = create_method(name).predict(observed, mask)
        assert predicted.shape == observed.shape
        np.testing.assert_allclose(predicted[mask], observed[mask], atol=2e-3)
        assert np.isfinite(predicted).all()


def test_mean_fill_is_per_channel():
    values = np.zeros((4, 4, 2))
    mask = np.zeros((4, 4, 1), dtype=bool)
    mask[0, 0] = mask[-1, -1] = True
    values[0, 0] = (2, 10)
    values[-1, -1] = (4, 20)
    result = create_method("mean_fill").predict(values, mask)
    np.testing.assert_allclose(result[1, 1], (3, 15))


def test_persistence_rollout_shape():
    final_state = np.arange(16).reshape(4, 4, 1)
    result = create_method("persistence").predict(final_state, horizon=4)
    assert result.shape == (4, 4, 4, 1)
    np.testing.assert_array_equal(result[3], final_state)


def test_external_method_decorator(monkeypatch):
    # Keep this plugin-registration test isolated from later tests that compare
    # generated website options with the built-in method registry.
    monkeypatch.setattr(method_base, "METHOD_REGISTRY", dict(method_base.METHOD_REGISTRY))
    monkeypatch.setattr(method_base, "_PRIMARY_NAMES", set(method_base._PRIMARY_NAMES))

    @register_method("test_external_method", replace=True)
    class External:
        name = "test_external_method"
        capabilities = MethodCapabilities(tasks=frozenset({"recovery"}))

        def predict(self, observations, mask=None, **kwargs):
            return observations

    assert "test_external_method" in available_methods()
    assert isinstance(create_method("test-external-method"), External)


def test_compact_neural_shapes_and_reference_label():
    torch = pytest.importorskip("torch")
    inputs = torch.randn(2, 1, 17, 19)
    mask = torch.ones(2, 1, 17, 19)
    for name in ("unet", "fno", "cno"):
        kwargs = {"width": 8}
        if name == "fno":
            kwargs.update(modes=3, layers=2)
        model = create_model(name, **kwargs)
        assert model(inputs, mask=mask).shape == inputs.shape
        assert model.capabilities.reference_only
        assert "not an exact" in model.capabilities.notes.lower()


def test_rollout_neural_shapes():
    torch = pytest.importorskip("torch")
    model = create_model("convlstm", hidden_channels=8)
    result = model(torch.randn(2, 2, 1, 8, 8), horizon=3)
    assert result.shape == (2, 3, 1, 8, 8)


def test_residual_encoder_multitask_shapes_and_registry():
    torch = pytest.importorskip("torch")
    inputs = torch.randn(3, 1, 17, 19)
    mask = torch.ones(3, 1, 17, 19)
    model = create_model(
        "resnet_encoder",
        width=4,
        embedding_dim=12,
        class_counts={"family": 7, "boundary": 4, "setting": 10, "regime": 3},
    )

    embedding = model(inputs, mask=mask)
    assert embedding.shape == (3, 12)
    torch.testing.assert_close(torch.linalg.vector_norm(embedding, dim=1), torch.ones(3))
    logits = model(inputs, mask=mask, output="logits")
    assert {name: tuple(value.shape) for name, value in logits.items()} == {
        "family": (3, 7),
        "boundary": (3, 4),
        "setting": (3, 10),
        "regime": (3, 3),
    }
    targets = {name: torch.zeros(3, dtype=torch.long) for name in logits}
    loss = model.supervised_loss(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()

    discovered = create_method("supervised-multitask-small", width=4, embedding_dim=8)
    assert discovered.capabilities.supports("retrieval")
    assert discovered.capabilities.supports("supervised_multitask")
    assert discovered.capabilities.reference_only
    assert "not an exact" in discovered.capabilities.notes.lower()
    assert "residual_cnn" in available_methods()
    assert "residual_cnn" in PROJECT_METHOD_REGISTRY.names()


def test_mae_small_masks_reconstructs_and_preserves_visible_values():
    torch = pytest.importorskip("torch")
    inputs = torch.randn(2, 1, 17, 19)
    mask = torch.ones(2, 1, 17, 19)
    mask[:, :, :2] = 0
    model = create_method(
        "masked-autoencoder-small",
        width=4,
        latent_channels=8,
        patch_size=4,
        mask_ratio=0.5,
    )

    model.eval()
    reconstruction = model(inputs, mask=mask)
    assert reconstruction.shape == inputs.shape
    torch.testing.assert_close(reconstruction[mask.bool()], inputs[mask.bool()])
    assert model.embedding(inputs, mask=mask).shape == (2, 8)

    torch.manual_seed(7)
    model.train()
    masked, visible = model.mask_inputs(inputs, mask=mask, force_random=True)
    assert masked.shape == inputs.shape
    assert visible.shape == (2, 1, 17, 19)
    assert torch.all(visible <= mask)
    assert torch.count_nonzero(visible) < torch.count_nonzero(mask)
    assert torch.count_nonzero(masked * (1 - visible)) == 0
    training_reconstruction = model(inputs, mask=mask)
    assert training_reconstruction.shape == inputs.shape

    hidden_loss = model.reconstruction_loss(training_reconstruction, inputs, visible)
    assert hidden_loss.ndim == 0 and torch.isfinite(hidden_loss)
    assert model.capabilities.supports("pretraining")
    assert model.capabilities.supports("inverse")
    assert model.capabilities.reference_only
    assert "not a vit-mae" in model.capabilities.notes.lower()
    assert "mae_small" in available_methods()
    assert "mae_small" in PROJECT_METHOD_REGISTRY.names()
