from __future__ import annotations

import torch
from torch import nn

from .policy import PolicyNetwork
from .vision import VisionNetwork


class VisionPolicyModel(nn.Module):
    def __init__(self, vision: VisionNetwork, policy: PolicyNetwork):
        super().__init__()
        # 画像特徴を抽出するVision Network
        self.vision = vision
        # 画像特徴と状態から行動を出すPolicy Network
        self.policy = policy

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 画像列から特徴量を抽出(I_tをi_tに変換)
        feature = self.vision(image)
        # 特徴量とstateを使って行動を予測
        # PolicyNetworkに特徴量i_tとfollowerの関節q_tを渡し、Leaderの関節a_tを予測
        return self.policy(feature, state, h)

    def forward_step(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # image/state: [B, 1, ...] -> action_hat: [B, 1, Da]
        feature = self.vision(image)
        return self.policy.forward_step(feature, state, h)
