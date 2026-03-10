from __future__ import annotations

import torch
from torch import nn


class VisionCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_channels: list[int],
        kernels: list[int],
        strides: list[int],
        paddings: list[int],
        feature_dim: int,
    ):
        super().__init__()
        if not (len(conv_channels) == len(kernels) == len(strides) == len(paddings)):
            raise ValueError("conv setting lengths must match")

        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out, k, s, p in zip(conv_channels, kernels, strides, paddings):
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p))
            layers.append(nn.ReLU())
            c_in = c_out
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(conv_channels[-1], feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: [B, S, C, H, W] -> feature: [B, S, F]
        bsz, seq, c, h, w = image.shape
        x = image.reshape(bsz * seq, c, h, w)
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.proj(x)
        return x.reshape(bsz, seq, -1)
