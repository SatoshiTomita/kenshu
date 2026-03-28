from __future__ import annotations

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        rnn_type: str = "rnn",
        action_horizon: int = 1,
    ):
        super().__init__()
        self.action_horizon = int(action_horizon)
        rnn_type = rnn_type.lower()
        if rnn_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")
        # 出力ヘッドの次元を拡張
        self.head = nn.Linear(hidden_dim, output_dim * max(self.action_horizon, 1))

    def forward(
        self,
        features: torch.Tensor,
        state: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # features: [B, S, F], state: [B, S, Dq] -> action_hat: [B, S, Da]
        x = torch.cat([features, state], dim=-1)
        out, h_next = self.rnn(x, h)
        pred = self.head(out)
        if self.action_horizon > 1:
            # 出力を[B,T,K,Daに整形]
            b, t, _ = pred.shape
            pred = pred.view(b, t, self.action_horizon, -1)
        return pred, h_next

    def forward_step(
        self,
        features: torch.Tensor,
        state: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # features/state: [B, 1, *] -> action_hat: [B, 1, Da]
        x = torch.cat([features, state], dim=-1)
        out, h_next = self.rnn(x, h)
        pred = self.head(out)
        if self.action_horizon > 1:
            b, t, _ = pred.shape
            pred = pred.view(b, t, self.action_horizon, -1)
        return pred, h_next
