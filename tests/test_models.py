"""Tests for the shared models (src/models). Run: python -m pytest tests/test_models.py -q"""
import pytest
import torch

from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import CorruptionClassifier

X = torch.rand(2, 3, 128, 128)


def test_autoencoder_shapes_and_range():
    model = ConvAutoencoder(base_channels=16, depth=4, latent_channels=32).eval()
    with torch.no_grad():
        z, feats = model.encode(X)
        out = model(X)
    assert z.shape == (2, 32, 8, 8)
    assert model.latent_dim == 32 * 8 * 8
    assert [f.shape[-1] for f in feats] == [128, 64, 32, 16, 8]
    assert out.shape == X.shape and out.min() >= 0 and out.max() <= 1


def test_autoencoder_is_a_real_bottleneck():
    model = ConvAutoencoder(base_channels=16, latent_channels=32)
    assert 3 * 128 * 128 / model.latent_dim == 24  # compression ratio of the latent
    # Without skips the output depends on the input only through the latent
    model.eval()
    with torch.no_grad():
        z, _ = model.encode(X)
        assert torch.allclose(model.decode(z), model(X))


@pytest.mark.parametrize("skips", [(16,), (32, 16), (128,)])
def test_autoencoder_limited_skips(skips):
    model = ConvAutoencoder(base_channels=8, latent_channels=8, skip_resolutions=skips).eval()
    with torch.no_grad():
        assert model(X).shape == X.shape
    assert model.config["skip_resolutions"] == skips


def test_autoencoder_rejects_invalid_skip():
    with pytest.raises(ValueError):
        ConvAutoencoder(skip_resolutions=(8,))  # the latent resolution itself is not a skip


def test_autoencoder_dropout_only_in_training():
    model = ConvAutoencoder(base_channels=8, latent_channels=8, dropout=0.5)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(X), model(X))


def test_rebuild_from_config():
    for model in (ConvAutoencoder(base_channels=8, latent_channels=4), CorruptionClassifier(channels=(8, 16, 32))):
        clone = type(model)(**model.config)
        clone.load_state_dict(model.state_dict())


def test_classifier_logits():
    model = CorruptionClassifier(channels=(16, 32, 64), dropout=0.2).eval()
    with torch.no_grad():
        logits = model(X)
    assert logits.shape == (2, 4)
    loss = torch.nn.functional.cross_entropy(model.train()(X), torch.tensor([0, 3]))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
