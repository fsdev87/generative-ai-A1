"""Corruption-type classifier (Task 2), reused as the gating network of the soft MoE (Task 3)."""
import torch
import torch.nn as nn


class CorruptionClassifier(nn.Module):
    """CNN mapping an RGB image to logits over CLASSES = (clean, salt, blur, occlusion).

    Each stage is `convs_per_stage` 3x3 conv-BN-ReLU layers followed by 2x2 max pooling.
    The first stage runs at full resolution because the cues are high-frequency:
    salt-and-pepper pixels and mild blur would be destroyed by early downsampling.
    The head concatenates global average and global max pooling, since the cues are
    sparse (a few isolated pixels, one black rectangle) and averaging alone dilutes them.
    """

    def __init__(self, channels=(32, 64, 128, 256), convs_per_stage=2, dropout=0.3, num_classes=4):
        super().__init__()
        self.config = dict(channels=tuple(channels), convs_per_stage=convs_per_stage,
                           dropout=dropout, num_classes=num_classes)
        layers, in_ch = [], 3
        for out_ch in channels:
            for _ in range(convs_per_stage):
                layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
                in_ch = out_ch
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * channels[-1], num_classes))

    def forward(self, x):
        h = self.features(x)
        return self.head(torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1))
