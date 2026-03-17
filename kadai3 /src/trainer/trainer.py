from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        state_noise_std: float = 0.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.state_noise_std = float(state_noise_std)
        self.loss_fn = nn.MSELoss()

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        # 学習/評価で共通の1epoch処理（train=Trueなら勾配更新）
        self.model.train(train)
        total = 0.0
        count = 0
        for image, state, action in loader:
            image = image.to(self.device)
            state = state.to(self.device)
            action = action.to(self.device)
            if self.state_noise_std > 0.0 and train:
                # stateにノイズを加えてロバスト性を上げる
                noise = torch.randn_like(state) * self.state_noise_std
                state = state + noise
            pred = self.model(image, state)
            loss = self.loss_fn(pred, action)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total += float(loss.item())
            count += 1
        return total / max(count, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> list[dict[str, float]]:
        # 学習と検証をepochs回まわし、lossログを返す
        history: list[dict[str, float]] = []
        for epoch in range(int(epochs)):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        return history
