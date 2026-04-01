from __future__ import annotations

import math
import torch
from torch import nn


def get_conved_size(
    obs_size: list[int],
    channels: list[int],
    kernels: list[int],
    strides: list[int],
    paddings: list[int],
) -> int:
    if len(obs_size) != 3:
        raise ValueError(f"obs_size must be [C,H,W], got: {obs_size}")
    if not (len(channels) >= 2 and len(channels) - 1 == len(kernels) == len(strides) == len(paddings)):
        raise ValueError("conv setting lengths must satisfy len(channels)-1 == len(kernels)==len(strides)==len(paddings)")
    _, h, w = obs_size
    for k, s, p in zip(kernels, strides, paddings):
        h = (h + 2 * p - k) // s + 1
        w = (w + 2 * p - k) // s + 1
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid conv output size: ({h}, {w})")
    return channels[-1] * h * w


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_layers: int,
) -> nn.Module:
    n_layers = max(int(n_layers), 1)
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.ReLU())
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(
        self,
        channels: list[int],
        kernels: list[int],
        strides: list[int],
        paddings: list[int],
        latent_obs_dim: int,
        mlp_hidden_dim: int,
        n_mlp_layers: int,
    ):
        super().__init__()
        obs_size = [3, 48, 64]
        self.encoder = self._build_conv_layers(channels, kernels, strides, paddings)
        self.fc = _build_mlp(
            get_conved_size(obs_size, channels, kernels, strides, paddings),
            latent_obs_dim * 2,
            mlp_hidden_dim,
            n_mlp_layers,
        )

    @staticmethod
    def _build_conv_layers(
        channels: list[int],
        kernels: list[int],
        strides: list[int],
        paddings: list[int],
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        for c_in, c_out, k, s, p in zip(channels[:-1], channels[1:], kernels, strides, paddings):
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


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

        self.pool_type = pool_type.lower()
        if self.pool_type not in ("avg", "spatial_softmax"):
            raise ValueError(f"Unsupported pool_type: {pool_type}. Use 'avg' or 'spatial_softmax'.")

        channels = [in_channels] + list(conv_channels)
        latent_obs_dim = int(math.ceil(float(feature_dim) / 2.0))
        self.encoder = Encoder(
            channels=channels,
            kernels=list(kernels),
            strides=list(strides),
            paddings=list(paddings),
            latent_obs_dim=latent_obs_dim,
            mlp_hidden_dim=feature_dim,
            n_mlp_layers=2,
        )
        enc_out_dim = latent_obs_dim * 2
        self.proj = nn.Identity() if enc_out_dim == feature_dim else nn.Linear(enc_out_dim, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: [B, S, C, H, W] -> feature: [B, S, F]
        bsz, seq, c, h, w = image.shape
        x = image.reshape(bsz * seq, c, h, w)
        x = self.encoder(x)
        x = self.proj(x)
        return x.reshape(bsz, seq, -1)
