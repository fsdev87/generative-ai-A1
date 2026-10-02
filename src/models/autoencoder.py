"""Convolutional autoencoder shared by Task 1 (universal model), Task 2 (specialists) and Task 3 (experts)."""
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, stride=1, dropout=0.0):
    """Two 3x3 conv-BN-ReLU layers; the first one downsamples when stride=2."""
    layers = [
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class ConvAutoencoder(nn.Module):
    """Encoder -> compressed latent -> decoder for RGB images in [0, 1].

    The encoder halves the resolution `depth` times with strided convolutions while
    doubling the channels (capped at 8 x base_channels). A 1x1 convolution projects the
    deepest features to `latent_channels`, so the latent holds
    latent_channels x (image_size / 2**depth)**2 values. That latent is the only path
    from input to output unless `skip_resolutions` adds limited skip connections:
    e.g. (16,) concatenates the 16x16 encoder features into the decoder. The decoder
    upsamples with bilinear resize + convolution, which avoids the checkerboard
    artifacts of transposed convolutions, and ends in a sigmoid.
    """

    def __init__(self, base_channels=32, depth=4, latent_channels=32, dropout=0.0,
                 skip_resolutions=(), image_size=128):
        super().__init__()
        self.config = dict(base_channels=base_channels, depth=depth, latent_channels=latent_channels,
                           dropout=dropout, skip_resolutions=tuple(skip_resolutions), image_size=image_size)
        self.depth = depth
        self.resolutions = [image_size // 2**i for i in range(depth + 1)]
        self.skip_resolutions = tuple(skip_resolutions)
        invalid = set(self.skip_resolutions) - set(self.resolutions[:-1])
        if invalid:
            raise ValueError(f"skip resolutions {sorted(invalid)} not in encoder resolutions {self.resolutions[:-1]}")
        chans = [base_channels * min(2**i, 8) for i in range(depth + 1)]

        self.stem = conv_block(3, chans[0], dropout=dropout)
        self.down = nn.ModuleList(conv_block(chans[i], chans[i + 1], stride=2, dropout=dropout) for i in range(depth))
        self.to_latent = nn.Conv2d(chans[-1], latent_channels, 1)
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_channels, chans[-1], 1, bias=False), nn.BatchNorm2d(chans[-1]), nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up = nn.ModuleList()
        for i in reversed(range(depth)):
            skip_ch = chans[i] if self.resolutions[i] in self.skip_resolutions else 0
            self.up.append(conv_block(chans[i + 1] + skip_ch, chans[i], dropout=dropout))
        self.head = nn.Sequential(nn.Conv2d(chans[0], 3, 3, padding=1), nn.Sigmoid())

    @property
    def latent_dim(self):
        return self.config["latent_channels"] * self.resolutions[-1] ** 2

    def encode(self, x):
        """Return the latent and the encoder features (index i has resolution image_size / 2**i)."""
        feats = [self.stem(x)]
        for block in self.down:
            feats.append(block(feats[-1]))
        return self.to_latent(feats[-1]), feats

    def decode(self, z, feats=None):
        h = self.from_latent(z)
        for i, block in zip(reversed(range(self.depth)), self.up):
            h = self.upsample(h)
            if self.resolutions[i] in self.skip_resolutions:
                h = torch.cat([h, feats[i]], dim=1)
            h = block(h)
        return self.head(h)

    def forward(self, x):
        z, feats = self.encode(x)
        return self.decode(z, feats)
