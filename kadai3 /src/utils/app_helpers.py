from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.episode_dataset import Normalizer
from src.models.policy import PolicyNetwork
from src.models.vision import VisionNetwork
from src.models.vision_policy import VisionPolicyModel


def build_model(cfg, state_dim: int, action_dim: int) -> nn.Module:
    vision = VisionNetwork(
        in_channels=cfg.vision.in_channels,
        conv_channels=list(cfg.vision.conv_channels),
        kernels=list(cfg.vision.kernels),
        strides=list(cfg.vision.strides),
        paddings=list(cfg.vision.paddings),
        feature_dim=cfg.vision.feature_dim,
    )
    policy = PolicyNetwork(
        input_dim=cfg.vision.feature_dim + state_dim,
        hidden_dim=cfg.policy.hidden_dim,
        output_dim=action_dim,
        num_layers=cfg.policy.num_layers,
        rnn_type=cfg.policy.rnn_type,
    )
    return VisionPolicyModel(vision=vision, policy=policy)


def _moving_avg(x: np.ndarray, win: int = 5) -> np.ndarray:
    if x.size == 0:
        return x
    win = max(int(win), 1)
    if win <= 1:
        return x
    pad = win // 2
    x_pad = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    out = np.zeros_like(x, dtype=np.float32)
    for d in range(x.shape[1]):
        out[:, d] = np.convolve(x_pad[:, d], kernel, mode="valid")
    return out


def save_action_figs(pred_all: np.ndarray, gt_all: np.ndarray | None, fig_dir: Path, prefix: str) -> None:
    import matplotlib.pyplot as plt

    if pred_all.size == 0:
        return
    pred_s = _moving_avg(pred_all, win=5)
    gt_s = _moving_avg(gt_all, win=5) if gt_all is not None else None
    n_dims = pred_s.shape[1]

    for d in range(n_dims):
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.plot(pred_s[:, d], label="pred", linewidth=1.0)
        if gt_s is not None:
            ax.plot(gt_s[:, d], label="follower", linewidth=1.0)
        ax.set_title(f"{prefix} action dim {d} (smoothed)")
        ax.legend()
        fig.savefig(fig_dir / f"{prefix}_action_dim_{d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    ncols = 2
    nrows = (n_dims + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows), squeeze=False)
    for d in range(n_dims):
        r, c = divmod(d, ncols)
        ax = axes[r][c]
        ax.plot(pred_s[:, d], label="pred", linewidth=1.0)
        if gt_s is not None:
            ax.plot(gt_s[:, d], label="follower", linewidth=1.0)
        ax.set_title(f"Action dim {d} (smoothed)")
        ax.legend()
    for idx in range(n_dims, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_action_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def offline_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_norm: Normalizer,
    fig_dir: Path,
    prefix: str = "offline",
) -> dict:
    model.eval()
    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    mse_values: list[float] = []

    with torch.no_grad():
        for image, state, action in loader:
            image = image.to(device)
            state = state.to(device)
            action = action.to(device)
            pred = model(image, state)

            pred_np = pred.detach().cpu().numpy().reshape(-1, pred.shape[-1])
            gt_np = action.detach().cpu().numpy().reshape(-1, action.shape[-1])

            pred_denorm = action_norm.denormalize(pred_np)
            gt_denorm = action_norm.denormalize(gt_np)
            mse_values.append(float(np.mean((pred_denorm - gt_denorm) ** 2)))

            all_pred.append(pred_denorm)
            all_gt.append(gt_denorm)

    pred_all = np.concatenate(all_pred, axis=0) if all_pred else np.zeros((0, 1), dtype=np.float32)
    gt_all = np.concatenate(all_gt, axis=0) if all_gt else np.zeros((0, 1), dtype=np.float32)
    mse = float(np.mean(mse_values)) if mse_values else float("nan")

    save_action_figs(pred_all=pred_all, gt_all=gt_all, fig_dir=fig_dir, prefix=prefix)
    return {f"{prefix}_mse": mse, "num_samples": int(len(pred_all))}


def run_replay(cfg, model: nn.Module, state_norm: Normalizer, action_norm: Normalizer, device: torch.device, fig_dir: Path):
    from lerobot_utils import Replay  # type: ignore

    replay = Replay(
        height=480,
        width=640,
        camera_id=(0,),
        is_higher_port=False,
        leader_port="/dev/tty.usbmodem58370530001",
        follower_port="/dev/tty.usbmodem58370529971",
    )

    image_q: list[torch.Tensor] = []
    state_q: list[np.ndarray] = []
    pred_actions: list[np.ndarray] = []
    model.eval()
    for _ in range(int(cfg.replay.steps)):
        obs = replay.get_observations(max_depth=cfg.replay.max_depth)
        image = obs[cfg.replay.image_key]
        state = np.asarray(obs[cfg.replay.state_key], dtype=np.float32)

        image_t = torch.tensor(np.asarray(image), dtype=torch.float32)
        if image_t.ndim == 3 and image_t.shape[0] not in (1, 3) and image_t.shape[-1] in (1, 3):
            image_t = image_t.permute(2, 0, 1)
        if image_t.max() > 1.0:
            image_t = image_t / 255.0
        if cfg.replay.resize_height is not None and cfg.replay.resize_width is not None:
            image_t = F.interpolate(
                image_t.unsqueeze(0),
                size=(int(cfg.replay.resize_height), int(cfg.replay.resize_width)),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        image_q.append(image_t)
        state_q.append(state)
        if len(image_q) < int(cfg.dataset.seq_len):
            continue
        image_q = image_q[-int(cfg.dataset.seq_len) :]
        state_q = state_q[-int(cfg.dataset.seq_len) :]

        image_in = torch.stack(image_q, dim=0).unsqueeze(0).to(device)
        state_np = np.stack(state_q, axis=0)
        state_np = state_norm.normalize(state_np)
        state_in = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_normed = model(image_in, state_in)[0, -1].detach().cpu().numpy()
        action = action_norm.denormalize(action_normed)
        action_t = torch.tensor(action, dtype=torch.float32)
        pred_actions.append(action)
        if cfg.replay.send_action:
            replay.send(
                action=action_t,
                fps=cfg.replay.fps,
            )

    if pred_actions:
        save_action_figs(pred_all=np.asarray(pred_actions, dtype=np.float32), gt_all=None, fig_dir=fig_dir, prefix="online")
    return {"replay_steps": int(cfg.replay.steps), "send_action": bool(cfg.replay.send_action)}
