from __future__ import annotations

from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.episode_dataset import Normalizer


def run_online_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_norm: Normalizer,
    fig_dir: Path,
) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
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

    for d in range(min(pred_all.shape[1], gt_all.shape[1])):
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.plot(gt_all[:, d], label="gt")
        ax.plot(pred_all[:, d], label="pred")
        ax.set_title(f"Online test action dim {d}")
        ax.legend()
        fig.savefig(fig_dir / f"online_action_dim_{d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {"online_mse": mse, "num_samples": int(len(pred_all))}


def _to_tensor_image(
    image: object,
    device: torch.device,
    resize_height: int | None,
    resize_width: int | None,
) -> torch.Tensor:
    if torch.is_tensor(image):
        t = image.detach().clone().float()
    else:
        t = torch.tensor(np.asarray(image), dtype=torch.float32)

    if t.ndim == 3 and t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)
    if t.max() > 1.0:
        t = t / 255.0
    if resize_height is not None and resize_width is not None:
        t = F.interpolate(
            t.unsqueeze(0),
            size=(int(resize_height), int(resize_width)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return t.to(device)


def run_offline_replay(
    cfg,
    model: torch.nn.Module,
    state_norm: Normalizer,
    action_norm: Normalizer,
    device: torch.device,
    seq_len: int,
):
    from lerobot_utils import Replay

    replay = Replay(
        height=cfg.replay.height,
        width=cfg.replay.width,
        serial_numbers=tuple(cfg.replay.serial_numbers),
        camera_id=tuple(cfg.replay.camera_id),
        scale=cfg.replay.scale,
        fps=cfg.replay.fps,
        auto_exposure=cfg.replay.auto_exposure,
        auto_white_balance=cfg.replay.auto_white_balance,
        exposure=cfg.replay.exposure,
        white_balance=cfg.replay.white_balance,
        max_depth=cfg.replay.max_depth_realsense,
        min_depth=cfg.replay.min_depth,
        is_higher_port=cfg.replay.is_higher_port,
        leader_port=cfg.replay.leader_port,
        follower_port=cfg.replay.follower_port,
        calibration_name=cfg.replay.calibration_name,
    )

    model.eval()
    image_q: deque[torch.Tensor] = deque(maxlen=seq_len)
    state_q: deque[np.ndarray] = deque(maxlen=seq_len)

    for _ in range(cfg.replay.steps):
        obs = replay.get_observations(max_depth=cfg.replay.max_depth)
        image = _to_tensor_image(
            obs[cfg.replay.image_key],
            device=device,
            resize_height=cfg.replay.resize_height,
            resize_width=cfg.replay.resize_width,
        )
        state = np.asarray(obs[cfg.replay.state_key], dtype=np.float32)

        image_q.append(image)
        state_q.append(state)
        if len(image_q) < seq_len:
            continue

        image_in = torch.stack(list(image_q), dim=0).unsqueeze(0)
        state_np = np.stack(list(state_q), axis=0)
        state_np = state_norm.normalize(state_np)
        state_in = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_normed = model(image_in, state_in)[0, -1].detach().cpu().numpy()
        action = action_norm.denormalize(action_normed)
        action_t = torch.tensor(action, dtype=torch.float32)

        if cfg.replay.send_action:
            replay.send(
                action=action_t,
                fps=cfg.replay.fps,
                split=cfg.replay.split,
                ema=cfg.replay.ema,
            )

    return {"replay_steps": int(cfg.replay.steps), "send_action": bool(cfg.replay.send_action)}
