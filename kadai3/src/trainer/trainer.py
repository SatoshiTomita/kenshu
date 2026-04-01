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
        state_dropout: float = 0.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.state_noise_std = float(state_noise_std)
        self.state_dropout = float(state_dropout)
        self.loss_fn = nn.MSELoss()
        self._logged_shapes = False

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
                # followerのデータにノイズを加えてロバスト性を上げる
                noise = torch.randn_like(state) * self.state_noise_std
                state = state + noise
            if self.state_dropout > 0.0 and train:
                # state の一部をゼロ化して依存を弱める
                state = nn.functional.dropout(state, p=self.state_dropout, training=True)
            pred, _ = self.model(image, state)
            # Shapes: image/state/action = [B, T, C, H, W] / [B, T, D] / [B, T, Da]
            # pred = [B, T, Da] or [B, T, K, Da] (K=action_horizon)
            # Daは関節の次元数
            if pred.ndim == 4:
                # pred: [B,T,K,Da], action: [B,T,Da]なので形が合わない。ゆえに、actionをK個にずらして積み重ねる
                b, t, k, da = pred.shape
                if t < k:
                    raise RuntimeError("Sequence length is shorter than action_horizon.")
                # targetの時間長を (t-k) に合わせる
                target = torch.stack([action[:, i + 1 : i + 1 + (t - k), :] for i in range(k)], dim=2)
                pred_use = pred[:, : t - k , :, :]
                loss_action = self.loss_fn(pred_use, target)
            else:
                if action.shape[1] < 2:
                    raise RuntimeError("Sequence length must be >= 2 for 1-step-ahead target.")
                loss_action = self.loss_fn(pred[:, :-1, :], action[:, 1:, :])
            loss = loss_action
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if not self._logged_shapes:
                self._logged_shapes = True
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
