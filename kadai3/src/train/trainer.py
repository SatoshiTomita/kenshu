from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


@dataclass
class TrainLog:
    train_loss: float
    val_loss: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        save_dir: Path,
        use_wandb: bool,
        grad_clip_norm: float | None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        self.grad_clip_norm = grad_clip_norm
        self.criterion = nn.MSELoss()

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], train: bool) -> float:
        image, state, action = batch
        image = image.to(self.device)
        state = state.to(self.device)
        action = action.to(self.device)

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        pred = self.model(image, state)
        loss = self.criterion(pred, action)

        if train:
            loss.backward()
            if self.grad_clip_norm and self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

        return float(loss.detach().cpu().item())

    def _run_loader(self, loader: DataLoader, train: bool) -> float:
        if train:
            self.model.train()
        else:
            self.model.eval()

        losses: list[float] = []
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                losses.append(self._step(batch, train=train))
        return sum(losses) / max(len(losses), 1)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> list[TrainLog]:
        logs: list[TrainLog] = []
        best_val = float("inf")

        for epoch in tqdm(range(epochs), desc="train"):
            train_loss = self._run_loader(train_loader, train=True)
            val_loss = self._run_loader(val_loader, train=False)
            logs.append(TrainLog(train_loss=train_loss, val_loss=val_loss))

            if self.use_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), self.save_dir / "best_model.pt")
                save_file(self.model.state_dict(), str(self.save_dir / "best_model.safetensors"))

        return logs
