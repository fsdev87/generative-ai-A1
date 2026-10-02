"""Task 4 conditional GAN: style-conditioned U-Net generator and PatchGAN discriminator.

G(x, s): pix2pix U-Net (Isola et al., 2017) for 128x128 photos. The encoder does not see
the style; every decoder layer is a conditional instance norm whose scale and shift are
linear projections of a learned style embedding (FiLM / conditional instance norm).
Output: tanh sketch in [-1, 1] with `out_channels` channels.

D(x, y, s): pix2pix 70x70 PatchGAN on concat(photo, sketch). Every patch logit is the
usual unconditional output plus a projection term (Miyato & Koyama, 2018): the inner
product between the patch's features and D's own style embedding mapped into feature space.

Both networks store their constructor arguments in `self.config` (rebuild with
Model(**config)). Alternatives and references: docs/fs2k_notes.md, "Design decisions".
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

INIT_STD = 0.02  # pix2pix / DCGAN Gaussian initialisation


def init_weights(model, std=INIT_STD):
    """pix2pix initialisation: conv weights N(0, std), norm scales N(1, std), zero biases."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d) and m.affine:
            nn.init.normal_(m.weight, 1.0, std)
            nn.init.zeros_(m.bias)


class StyleFiLM(nn.Module):
    """Conditional instance norm: IN(h) * (1 + gamma(e)) + beta(e) for a style embedding e."""

    def __init__(self, channels, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        # Two projections instead of one Linear + chunk(): the dynamo ONNX exporter turns
        # chunk into an opset-18 Split that is invalid after conversion to opset 17.
        self.gamma = nn.Linear(style_dim, channels)
        self.beta = nn.Linear(style_dim, channels)
        for proj in (self.gamma, self.beta):
            # the embedding has unit variance, so gamma and beta start with std INIT_STD
            # for every style_dim: the decoder starts close to plain instance norm
            nn.init.normal_(proj.weight, 0.0, INIT_STD / math.sqrt(style_dim))
            nn.init.zeros_(proj.bias)

    def forward(self, h, e):
        return self.norm(h) * (1 + self.gamma(e)[:, :, None, None]) + self.beta(e)[:, :, None, None]


class UNetGenerator(nn.Module):
    """Style-conditioned U-Net G(photo, style) -> sketch for 128x128 inputs.

    Seven stride-2 4x4 convolutions take 128x128 to a 1x1 bottleneck (widths c, 2c, 4c,
    8c, 8c, 8c, 8c for base_channels c); six transposed convolutions with skip connections
    return to 64x64 and a last one to 128x128. Dropout follows the three innermost decoder
    layers, as in pix2pix, and is off in eval mode (deterministic inference and ONNX).
    """

    def __init__(self, in_channels=3, out_channels=1, base_channels=64, dropout=0.5,
                 style_dim=16, num_styles=3):
        super().__init__()
        self.config = dict(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels,
                           dropout=dropout, style_dim=style_dim, num_styles=num_styles)
        c = base_channels
        widths = [c, 2 * c, 4 * c, 8 * c, 8 * c, 8 * c, 8 * c]
        self.down = nn.ModuleList()
        prev = in_channels
        for i, width in enumerate(widths):
            outer, inner = i == 0, i == len(widths) - 1
            # no norm on the first layer (pix2pix) nor on the 1x1 bottleneck, where
            # instance norm would map every channel to zero
            layers = [] if outer else [nn.LeakyReLU(0.2)]
            layers.append(nn.Conv2d(prev, width, 4, 2, 1, bias=outer or inner))
            if not (outer or inner):
                layers.append(nn.InstanceNorm2d(width, affine=True))
            self.down.append(nn.Sequential(*layers))
            prev = width
        self.up, self.film = nn.ModuleList(), nn.ModuleList()
        for i, width in enumerate(widths[-2::-1]):  # 8c, 8c, 8c, 4c, 2c, c at 2, 4, ..., 64 px
            in_width = prev if i == 0 else 2 * prev  # later layers also get the encoder skip
            self.up.append(nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(in_width, width, 4, 2, 1, bias=False)))
            self.film.append(StyleFiLM(width, style_dim))
            prev = width
        self.out = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(2 * prev, out_channels, 4, 2, 1), nn.Tanh())
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(num_styles, style_dim)
        init_weights(self)

    def forward(self, photo, style):
        e = self.embed(style)
        skips, h = [], photo
        for down in self.down:
            h = down(h)
            skips.append(h)
        skips.pop()  # the bottleneck is h itself
        for i, (up, film) in enumerate(zip(self.up, self.film)):
            h = film(up(h if i == 0 else torch.cat([h, skips.pop()], 1)), e)
            if i < 3:
                h = self.dropout(h)
        return self.out(torch.cat([h, skips.pop()], 1))


class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN D(photo, sketch, style) -> patch logits (N, 1, 14, 14) for
    128x128 inputs and n_layers=3 (70x70 receptive field; n_layers=2: 30x30 map, 34x34 field).

    The output layer has a shared 4x4 kernel plus a style-specific 4x4 kernel generated
    from the style embedding: logit(p) = w.phi(p) + b + v(s).phi(p), where phi(p) is the
    window of last-layer features under patch p and v(s) = W e(s). This is the projection
    discriminator of Miyato & Koyama (2018) applied to every patch.
    """

    def __init__(self, photo_channels=3, sketch_channels=1, base_channels=64, n_layers=3,
                 style_dim=16, num_styles=3):
        super().__init__()
        self.config = dict(photo_channels=photo_channels, sketch_channels=sketch_channels,
                           base_channels=base_channels, n_layers=n_layers, style_dim=style_dim,
                           num_styles=num_styles)
        c = base_channels
        layers = [nn.Conv2d(photo_channels + sketch_channels, c, 4, 2, 1), nn.LeakyReLU(0.2)]
        prev = c
        for i in range(1, n_layers + 1):
            width = c * min(2 ** i, 8)
            layers += [
                nn.Conv2d(prev, width, 4, 2 if i < n_layers else 1, 1, bias=False),
                nn.InstanceNorm2d(width, affine=True),
                nn.LeakyReLU(0.2),
            ]
            prev = width
        self.features = nn.Sequential(*layers)
        self.out = nn.Conv2d(prev, 1, 4, 1, 1)
        self.embed = nn.Embedding(num_styles, style_dim)
        self.proj = nn.Linear(style_dim, prev * 16, bias=False)
        init_weights(self)
        # style kernels start on the same scale as the shared kernel (std INIT_STD)
        nn.init.normal_(self.proj.weight, 0.0, INIT_STD / math.sqrt(style_dim))

    def forward(self, photo, sketch, style):
        f = self.features(torch.cat([photo, sketch], 1))
        kernels = self.proj(self.embed.weight).view(-1, f.shape[1], 4, 4)  # one kernel per style
        per_style = F.conv2d(f, kernels, padding=1)  # (N, num_styles, h, w)
        index = style.view(-1, 1, 1, 1).expand(-1, 1, *per_style.shape[2:])
        return self.out(f) + per_style.gather(1, index)
