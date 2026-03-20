from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.episode_dataset import Normalizer
from src.utils.train_utils import load_episodes
from src.models.policy import PolicyNetwork
from src.models.vision import VisionNetwork
from src.models.vision_policy import VisionPolicyModel

# モデルの構築
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

# グラフを滑らかにする
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

# 各次元ごとに予想と実際の値をグラフに保存する
def save_action_figs(pred_all: np.ndarray, gt_all: np.ndarray | None, fig_dir: Path, prefix: str) -> None:
    import matplotlib.pyplot as plt

    if pred_all.size == 0:
        return
    pred_s = _moving_avg(pred_all, win=5)
    gt_s = _moving_avg(gt_all, win=5) if gt_all is not None else None
    if prefix == "online":
        pred_s = np.clip(pred_s, -1.0, 1.0)
        if gt_s is not None:
            gt_s = np.clip(gt_s, -1.0, 1.0)
    n_dims = pred_s.shape[1]

    for d in range(n_dims):
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.plot(pred_s[:, d], label="pred", linewidth=1.0)
        if gt_s is not None:
            ax.plot(gt_s[:, d], label="follower", linewidth=1.0)
        if prefix == "online":
            ax.set_ylim(-1.05, 1.05)
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
        if prefix == "online":
            ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"Action dim {d} (smoothed)")
        ax.legend()
    for idx in range(n_dims, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_action_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# オンラインテストの実行
def online_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_norm: Normalizer,
    fig_dir: Path,
    prefix: str = "online",
) -> dict:
    from PIL import Image

    model.eval()
    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    mse_values: list[float] = []
    saved_gif = False

    with torch.no_grad():
        for image, state, action in loader:
            if not saved_gif:
                seq = image[0].detach().cpu().numpy()  # [T,C,H,W]
                frames: list[Image.Image] = []
                for f in seq:
                    if f.shape[0] in (1, 3):
                        img = np.transpose(f, (1, 2, 0))
                    else:
                        img = f
                    img = np.clip(img, 0.0, 1.0)
                    img_u8 = (img * 255.0).astype(np.uint8)
                    frames.append(Image.fromarray(img_u8))
                if frames:
                    gif_path = fig_dir / f"{prefix}_camera.gif"
                    frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=frames[1:],
                        duration=100,
                        loop=0,
                    )
                    saved_gif = True
            image = image.to(device)
            state = state.to(device)
            action = action.to(device)
            pred, _, _ = model(image, state)

            pred_np = pred.detach().cpu().numpy().reshape(-1, pred.shape[-1])
            gt_np = action.detach().cpu().numpy().reshape(-1, action.shape[-1])

            # EpisodeDataset で既に正規化済みなので、そのまま比較・描画する
            mse_values.append(float(np.mean((pred_np - gt_np) ** 2)))

            all_pred.append(pred_np)
            all_gt.append(gt_np)

    pred_all = np.concatenate(all_pred, axis=0) if all_pred else np.zeros((0, 1), dtype=np.float32)
    gt_all = np.concatenate(all_gt, axis=0) if all_gt else np.zeros((0, 1), dtype=np.float32)
    mse = float(np.mean(mse_values)) if mse_values else float("nan")

    save_action_figs(pred_all=pred_all, gt_all=gt_all, fig_dir=fig_dir, prefix=prefix)
    return {f"{prefix}_mse": mse, "num_samples": int(len(pred_all))}

# 実機でのテストコード
def run_replay(cfg, model: nn.Module, state_norm: Normalizer, action_norm: Normalizer, device: torch.device, fig_dir: Path):
    from lerobot_utils import Replay  # type: ignore
    from PIL import Image, ImageDraw, ImageFont

    # --- 1. ハードウェアの初期化 ---
    # カメラやロボットアーム（USBポート）との接続を確立する
    replay = Replay(
        height=480,
        width=640,
        camera_id=(0,),
        is_higher_port=False,
        leader_port="/dev/tty.usbmodem58370530001",
        follower_port="/dev/tty.usbmodem58370529971",
    )

    # 過去のデータを溜めておくための「待ち行列（キュー）」を用意
    def _compute_init_pose() -> np.ndarray | None:
        # data の t=0 平均アクションを使う（アクション空間に合わせる）
        train_eps, test_eps = load_episodes(cfg)
        poses = [ep["action"][0] for ep in (train_eps + test_eps) if len(ep["action"]) > 0]
        if not poses:
            return None
        return np.mean(np.stack(poses, axis=0), axis=0).astype(np.float32)

    init_pose = _compute_init_pose()

    def _send_init_pose() -> None:
        if init_pose is None:
            return
        action_dim = int(action_norm.min.shape[0])
        pose = init_pose
        if pose.shape[0] != action_dim:
            if pose.shape[0] > action_dim:
                print(
                    {
                        "init_pose_dim": int(pose.shape[0]),
                        "action_dim": action_dim,
                        "init_pose": "truncate_to_action_dim",
                    }
                )
                pose = pose[:action_dim]
            else:
                print(
                    {
                        "init_pose_dim": int(pose.shape[0]),
                        "action_dim": action_dim,
                        "init_pose": "skip_send_dim_mismatch",
                    }
                )
                return
        action = torch.tensor(pose, dtype=torch.float32)
        repeat = int(getattr(cfg.replay, "init_pose_repeat", 1) or 1)
        for _ in range(max(repeat, 1)):
            if cfg.replay.send_action:
                replay.send(action=action, fps=cfg.replay.fps)
            time.sleep(0.2)

    pred_actions: list[np.ndarray] = []
    frames: list[Image.Image] = []

    print(
        {
            "init_pose_send": True,
            "repeat": int(getattr(cfg.replay, "init_pose_repeat", 1) or 1),
            "init_pose_mean": init_pose.tolist() if init_pose is not None else None,
        }
    )
    _send_init_pose()

    # モデルを評価モード（学習をしないモード）に切り替え
    model.eval()

    h: torch.Tensor | None = None
    # --- 2. リアルタイム制御ループ（指定ステップ数分繰り返す） ---
    for step_idx in range(int(cfg.replay.steps)):
        # カメラ映像とロボットの現在の関節角度（state）を取得
        obs = replay.get_observations(max_depth=cfg.replay.max_depth)
        image = obs[cfg.replay.image_key]
        state = np.asarray(obs[cfg.replay.state_key], dtype=np.float32)

        # --- 3. 画像の前処理（前段処理） ---
        # 取得した画像をPyTorch用のテンソル形式に変換
        image_t = torch.tensor(np.asarray(image), dtype=torch.float32)
        
        # [H, W, C] を [C, H, W]（チャンネルを先頭に）並び替え
        if image_t.ndim == 3 and image_t.shape[0] not in (1, 3) and image_t.shape[-1] in (1, 3):
            image_t = image_t.permute(2, 0, 1)
        
        # 0~255のピクセル値を 0.0~1.0 にスケーリング
        if image_t.max() > 1.0:
            image_t = image_t / 255.0
        
        # 学習時と同じサイズ（例：48x64など）にリサイズ
        if cfg.replay.resize_height is not None and cfg.replay.resize_width is not None:
            image_t = F.interpolate(
                image_t.unsqueeze(0),
                size=(int(cfg.replay.resize_height), int(cfg.replay.resize_width)),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # GIF用に実機カメラの元画像を保存（色とスケールを補正）
        try:
            img = np.asarray(image)
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = np.clip(img, 0.0, 255.0)
            else:
                img = np.clip(img, 0.0, 1.0) * 255.0
            img_u8 = img.astype(np.uint8)
            if img_u8.ndim == 2:
                frame = Image.fromarray(img_u8, mode="L").convert("RGB")
            else:
                frame = Image.fromarray(img_u8, mode="RGB")
            draw = ImageDraw.Draw(frame)
            text = f"step {step_idx + 1}"
            # 文字が見えやすいように背景を描画
            draw.rectangle((4, 4, 80, 18), fill=(0, 0, 0))
            draw.text((6, 6), text, fill=(255, 255, 255), font=ImageFont.load_default())
            frames.append(frame)
        except Exception:
            pass

        # --- 4. 1ステップ入力を作成（隠れ状態を使い回す） ---
        image_in = image_t.unsqueeze(0).unsqueeze(0).to(device)  # [B=1,S=1,C,H,W]
        state_np = state_norm.normalize(state[None, :])  # [1,D]
        state_in = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1,1,D]

        # --- 5. モデルによる推論（意思決定） ---
        with torch.no_grad():
            action_seq, h = model.forward_step(image_in, state_in, h)
            h = h.detach() if h is not None else None
            action_normed = action_seq[0, 0].detach().cpu().numpy()
        print(f"Action predicted (normalized): {action_normed}")
        
        # 正規化された数値を「実際のロボットの角度」に復元（逆正規化）
        action = action_norm.denormalize(action_normed)
        action_t = torch.tensor(action, dtype=torch.float32)
        
        # グラフ作成用に推論結果を記録（正規化スケールのまま）
        pred_actions.append(action_normed)

        # --- 6. ロボットへの指令送信 ---
        if cfg.replay.send_action:
            # 計算した目標角度を実機のモーターに送信して動かす
            replay.send(
                action=action_t,
                fps=cfg.replay.fps,
            )

    # 全ステップ終了後、アクションの推移グラフを保存して終了
    if pred_actions:
        save_action_figs(pred_all=np.asarray(pred_actions, dtype=np.float32), gt_all=None, fig_dir=fig_dir, prefix="offline")

    if frames:
        gif_path = fig_dir / "online_camera.gif"
        try:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )
            print({"gif_saved": str(gif_path), "frames": len(frames)})
        except Exception as e:
            print({"gif_save_error": str(e), "gif_path": str(gif_path), "frames": len(frames)})
    else:
        print({"gif_saved": False, "reason": "no_frames"})

    _send_init_pose()
    
    return {"replay_steps": int(cfg.replay.steps), "send_action": bool(cfg.replay.send_action)}
