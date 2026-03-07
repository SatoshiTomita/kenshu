from __future__ import annotations

import torch
from torch import nn

from src.model.vision import VisionCNN


class PolicyRNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        rnn_type: str = "rnn",
    ):
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # features: [B, S, F], state: [B, S, Dq] -> action_hat: [B, S, Da]
        x = torch.cat([features, state], dim=-1)
        out, _ = self.rnn(x)
        return self.head(out)


class VisionPolicyModel(nn.Module):
    def __init__(self, vision: VisionCNN, policy: PolicyRNN):
        super().__init__()
        self.vision = vision
        self.policy = policy

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        feature = self.vision(image)
        return self.policy(feature, state)
