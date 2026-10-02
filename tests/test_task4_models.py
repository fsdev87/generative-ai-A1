"""Task 4 cGAN models: shapes, style conditioning, config round trip, ONNX export parity.
Run: python -m pytest tests/test_task4_models.py -q
"""
import json

import numpy as np
import pytest
import torch

from src.data.fs2k import make_smoke_data
from src.task4.models import PatchDiscriminator, UNetGenerator

SMALL = dict(base_channels=8, style_dim=8)


def _photos(n, seed=0):
    return torch.rand(n, 3, 128, 128, generator=torch.Generator().manual_seed(seed)) * 2 - 1


def test_generator_shape_range_and_depth():
    g = UNetGenerator(**SMALL).eval()
    sizes = []
    g.down[-1].register_forward_hook(lambda m, i, o: sizes.append(o.shape[-2:]))
    with torch.no_grad():
        out = g(_photos(4), torch.tensor([0, 1, 2, 0]))
    assert out.shape == (4, 1, 128, 128) and out.dtype == torch.float32
    assert out.abs().max() <= 1
    assert len(g.down) == 7 and sizes[0] == (1, 1)  # pix2pix unet_128: 1x1 bottleneck
    rgb = UNetGenerator(out_channels=3, **SMALL)
    assert rgb(_photos(2), torch.tensor([0, 1])).shape == (2, 3, 128, 128)


def test_styles_change_generator_output():
    g = UNetGenerator(**SMALL).eval()
    with torch.no_grad():
        out = g(_photos(1).repeat(3, 1, 1, 1), torch.arange(3))
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        assert (out[a] - out[b]).abs().max() > 1e-3


def test_style_embeddings_receive_gradients():
    g, d = UNetGenerator(**SMALL), PatchDiscriminator(**SMALL)
    x, s = _photos(2), torch.tensor([0, 2])
    d(x, g(x, s), s).mean().backward()
    for net in (g, d):
        grad = net.embed.weight.grad.abs().sum(dim=1)
        assert grad[0] > 0 and grad[2] > 0
        assert grad[1] == 0  # style 1 is not in the batch


def test_dropout_only_in_train_mode():
    g = UNetGenerator(dropout=0.5, **SMALL)
    x, s = _photos(2), torch.tensor([0, 1])
    with torch.no_grad():
        g.train()
        assert not torch.allclose(g(x, s), g(x, s))
        g.eval()
        assert torch.equal(g(x, s), g(x, s))


def test_batch_of_one_in_train_mode():
    g, d = UNetGenerator(**SMALL).train(), PatchDiscriminator(**SMALL).train()
    x, s = _photos(1), torch.tensor([2])
    assert d(x, g(x, s), s).shape == (1, 1, 14, 14)


def test_discriminator_patch_shape_and_style():
    d = PatchDiscriminator(**SMALL).eval()
    x, y = _photos(3), _photos(3, seed=1)[:, :1]
    with torch.no_grad():
        out = d(x, y, torch.tensor([0, 1, 2]))
        swapped = d(x, y, torch.tensor([1, 2, 0]))
    assert out.shape == (3, 1, 14, 14)
    assert (out - swapped).abs().amax(dim=(1, 2, 3)).min() > 0
    assert PatchDiscriminator(n_layers=2, **SMALL)(x, y, torch.tensor([0, 1, 2])).shape == (3, 1, 30, 30)
    assert PatchDiscriminator(sketch_channels=3, **SMALL)(x, x, torch.tensor([0, 1, 2])).shape == (3, 1, 14, 14)


@pytest.mark.parametrize("cls,kwargs", [
    (UNetGenerator, dict(base_channels=8, dropout=0.3, style_dim=4)),
    (PatchDiscriminator, dict(base_channels=8, n_layers=2, style_dim=4)),
])
def test_config_round_trip(cls, kwargs):
    a = cls(**kwargs)
    b = cls(**json.loads(json.dumps(a.config)))
    b.load_state_dict(a.state_dict())
    assert b.config == a.config


@pytest.mark.parametrize("dynamo", [False, True])
def test_generator_onnx_export_parity(tmp_path, dynamo):
    """docs/CONVENTIONS.md contract: photo float32 [N,3,128,128], style int64 [N] -> sketch,
    opset 17, dynamic batch, ONNX Runtime within 1e-4 of PyTorch on >= 32 inputs."""
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    g = UNetGenerator(base_channels=16, style_dim=8).eval()
    path = tmp_path / "generator.onnx"
    example = (_photos(2), torch.tensor([0, 2]))
    if dynamo:
        batch = torch.export.Dim("batch")
        axes = dict(dynamic_shapes={"photo": {0: batch}, "style": {0: batch}})
    else:
        axes = dict(dynamic_axes={"photo": {0: "batch"}, "style": {0: "batch"}, "sketch": {0: "batch"}})
    torch.onnx.export(g, example, str(path), input_names=["photo", "style"], output_names=["sketch"],
                      opset_version=17, dynamo=dynamo, **axes)

    model = onnx.load(str(path))
    assert [o.version for o in model.opset_import if o.domain in ("", "ai.onnx")] == [17]
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [(i.name, i.type) for i in sess.get_inputs()] == [("photo", "tensor(float)"), ("style", "tensor(int64)")]
    assert [o.name for o in sess.get_outputs()] == ["sketch"]

    smoke = np.concatenate([d["photos"] for d in make_smoke_data().values()])[:32]
    photos = torch.from_numpy(smoke).permute(0, 3, 1, 2).float() / 127.5 - 1
    styles = torch.arange(32) % 3
    for n in (1, 32):
        out = sess.run(["sketch"], {"photo": photos[:n].numpy(), "style": styles[:n].numpy()})[0]
        with torch.no_grad():
            ref = g(photos[:n], styles[:n]).numpy()
        assert out.shape == (n, 1, 128, 128)
        assert np.abs(out - ref).max() < 1e-4
