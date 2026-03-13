from __future__ import annotations

import torch
from torch import nn

from .policy import PolicyNetwork
from .vision import VisionNetwork


class VisionPolicyModel(nn.Module):
    def __init__(self, vision: VisionNetwork, policy: PolicyNetwork):
        super().__init__()
        self.vision = vision
        self.policy = policy

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        feature = self.vision(image)
        return self.policy(feature, state)
