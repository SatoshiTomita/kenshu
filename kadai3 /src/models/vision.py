from __future__ import annotations

import torch
from torch import nn


class VisionNetwork(nn.Module):
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
        # 簡易デコーダ（再構成用）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(conv_channels[-1], conv_channels[-2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(conv_channels[-2], conv_channels[-3], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(conv_channels[-3], in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: [B, S, C, H, W] -> feature: [B, S, F]
        bsz, seq, c, h, w = image.shape
        x = image.reshape(bsz * seq, c, h, w)
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.proj(x)
        return x.reshape(bsz, seq, -1)

    def forward_with_recon(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # image: [B, S, C, H, W] -> feature: [B, S, F], recon: [B, S, C, H, W]
        bsz, seq, c, h, w = image.shape
        x = image.reshape(bsz * seq, c, h, w)
        feat_map = self.encoder(x)
        pooled = self.pool(feat_map).squeeze(-1).squeeze(-1)
        feat = self.proj(pooled).reshape(bsz, seq, -1)
        recon_flat = self.decoder(feat_map)
        if recon_flat.shape[-2:] != (h, w):
            recon_flat = torch.nn.functional.interpolate(
                recon_flat, size=(h, w), mode="bilinear", align_corners=False
            )
        recon = recon_flat.reshape(bsz, seq, c, h, w)
        return feat, recon
