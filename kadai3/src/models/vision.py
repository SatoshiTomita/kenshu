from __future__ import annotations

import torch
from torch import nn


class SpatialSoftmax(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, C*2] (expected x,y per channel)
        b, c, h, w = x.shape
        x_flat = x.reshape(b, c, h * w)
        probs = torch.softmax(x_flat, dim=-1)
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pos_x = xx.reshape(-1)
        pos_y = yy.reshape(-1)
        exp_x = torch.sum(probs * pos_x, dim=-1)
        exp_y = torch.sum(probs * pos_y, dim=-1)
        return torch.cat([exp_x, exp_y], dim=-1)


class VisionNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_channels: list[int],
        kernels: list[int],
        strides: list[int],
        paddings: list[int],
        feature_dim: int,
        pool_type: str = "avg",
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
        self.pool_type = pool_type.lower()
        if self.pool_type == "spatial_softmax":
            self.pool = SpatialSoftmax()
            proj_in = conv_channels[-1] * 2
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            proj_in = conv_channels[-1]
        self.proj = nn.Linear(proj_in, feature_dim)
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
        x = self.pool(x)
        if self.pool_type != "spatial_softmax":
            x = x.squeeze(-1).squeeze(-1)
        x = self.proj(x)
        return x.reshape(bsz, seq, -1)

    def forward_with_recon(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # image: [B, S, C, H, W] -> feature: [B, S, F], recon: [B, S, C, H, W]
        bsz, seq, c, h, w = image.shape
        x = image.reshape(bsz * seq, c, h, w)
        feat_map = self.encoder(x)
        pooled = self.pool(feat_map)
        if self.pool_type != "spatial_softmax":
            pooled = pooled.squeeze(-1).squeeze(-1)
        feat = self.proj(pooled).reshape(bsz, seq, -1)
        recon_flat = self.decoder(feat_map)
        if recon_flat.shape[-2:] != (h, w):
            recon_flat = torch.nn.functional.interpolate(
                recon_flat, size=(h, w), mode="bilinear", align_corners=False
            )
        recon = recon_flat.reshape(bsz, seq, c, h, w)
        return feat, recon
