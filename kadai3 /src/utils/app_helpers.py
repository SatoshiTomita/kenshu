from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.dataloader.episode_dataset import Normalizer
from src.utils.train_utils import load_episodes
from src.models.policy import PolicyNetwork
from src.models.vision import VisionNetwork
from src.models.vision_policy import VisionPolicyModel

def _ensure_fig_subdir(fig_dir: Path, name: str) -> Path:
    subdir = fig_dir / name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir

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
        action_horizon=getattr(cfg.policy, "action_horizon", 1),
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
def save_action_figs(
    pred_all: np.ndarray,
    gt_all: np.ndarray | None,
    fig_dir: Path,
    prefix: str,
    episode_lengths: list[int] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if pred_all.size == 0:
        return
    pred_s = _moving_avg(pred_all, win=5)
    gt_s = _moving_avg(gt_all, win=5) if gt_all is not None else None
    pred_eps: list[np.ndarray] | None = None
    gt_eps: list[np.ndarray] | None = None
    if episode_lengths:
        pred_eps = []
        gt_eps = [] if gt_s is not None else None
        start = 0
        for ln in episode_lengths:
            end = start + int(ln)
            pred_eps.append(pred_s[start:end])
            if gt_s is not None and gt_eps is not None:
                gt_eps.append(gt_s[start:end])
            start = end
    if prefix.startswith("online"):
        pred_s = np.clip(pred_s, -1.0, 1.0)
        if gt_s is not None:
            gt_s = np.clip(gt_s, -1.0, 1.0)
    n_dims = pred_s.shape[1]

    for d in range(n_dims):
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        if pred_eps is not None:
            for ep in pred_eps:
                ax.plot(np.arange(ep.shape[0]), ep[:, d], label="pred" if d == 0 else None, linewidth=1.0, alpha=0.8)
            if gt_eps is not None:
                for ep in gt_eps:
                    ax.plot(np.arange(ep.shape[0]), ep[:, d], label="follower" if d == 0 else None, linewidth=1.0, alpha=0.8)
        else:
            ax.plot(pred_s[:, d], label="pred", linewidth=1.0)
            if gt_s is not None:
                ax.plot(gt_s[:, d], label="follower", linewidth=1.0)
        if prefix.startswith("online"):
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
        if pred_eps is not None:
            for ep in pred_eps:
                ax.plot(np.arange(ep.shape[0]), ep[:, d], label="pred" if d == 0 else None, linewidth=1.0, alpha=0.8)
            if gt_eps is not None:
                for ep in gt_eps:
                    ax.plot(np.arange(ep.shape[0]), ep[:, d], label="follower" if d == 0 else None, linewidth=1.0, alpha=0.8)
        else:
            ax.plot(pred_s[:, d], label="pred", linewidth=1.0)
            if gt_s is not None:
                ax.plot(gt_s[:, d], label="follower", linewidth=1.0)
        if prefix.startswith("online"):
            ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"Action dim {d} (smoothed)")
        ax.legend()
    for idx in range(n_dims, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_action_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 実機推論の可視化用: 全次元を1枚に重ねて描画
    if prefix == "offline_replay":
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(111)
        for d in range(n_dims):
            ax.plot(pred_s[:, d], linewidth=1.0, label=f"pred {d}")
        if gt_s is not None:
            for d in range(n_dims):
                ax.plot(gt_s[:, d], linewidth=1.0, linestyle="--", label=f"state {d}")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title("offline_replay action (all dims)")
        ax.set_xlabel("t")
        ax.set_ylabel("value (normalized)")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{prefix}_action_all_overlay.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_chunked_action_overlay(
    pred_all: np.ndarray,
    gt_all: np.ndarray | None,
    chunk_starts: list[int],
    fig_dir: Path,
    prefix: str,
    title: str | None = None,
    label_gt: str = "state",
) -> None:
    import matplotlib.pyplot as plt

    if pred_all.size == 0:
        return
    pred_s = _moving_avg(pred_all, win=5)
    if prefix.startswith("online"):
        pred_s = np.clip(pred_s, -1.0, 1.0)
    n_dims = pred_s.shape[1]

    fig, axes = plt.subplots(nrows=n_dims, ncols=1, figsize=(12, 2 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]
    for d, ax in enumerate(axes):
        for i, start in enumerate(chunk_starts):
            end = chunk_starts[i + 1] if i + 1 < len(chunk_starts) else pred_s.shape[0]
            x = np.arange(start, end)
            ax.plot(x, pred_s[start:end, d], linewidth=1.0, alpha=0.8, label="pred" if i == 0 else None)
        if gt_all is not None:
            gt_s = _moving_avg(gt_all, win=5)
            if prefix.startswith("online"):
                gt_s = np.clip(gt_s, -1.0, 1.0)
            ax.plot(np.arange(gt_s.shape[0]), gt_s[:, d], linestyle="--", linewidth=1.0, label=label_gt)
        for x0 in chunk_starts:
            ax.axvline(x0, color="k", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(f"dim {d}")
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("t (observation index)")
    fig.suptitle(title or "offline_replay action (chunked)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_action_all_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_future_action_figs(
    future_preds: list[np.ndarray],
    leader_traj_eps: list[np.ndarray] | None,
    fig_dir: Path,
    prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    if not future_preds:
        return
    arr = np.stack(future_preds, axis=0)  # [T, K, Da]
    t_len, k_len, d_len = arr.shape
    fig, axes = plt.subplots(nrows=d_len, ncols=1, figsize=(12, 2 * d_len), sharex=True)
    if d_len == 1:
        axes = [axes]
    for d, ax in enumerate(axes):
        for k in range(k_len):
            # 各時刻tからkステップ先までの予測軌道を短い線で描く
            for t in range(t_len):
                xs = np.arange(t, t + k_len)
                ys = arr[t, :, d]
                ax.plot(xs, ys, linewidth=3.2, alpha=0.45, color="tab:blue")
        if leader_traj_eps:
            for i, ep in enumerate(leader_traj_eps):
                if ep.shape[1] <= d:
                    continue
                x = np.arange(min(ep.shape[0], t_len))
                ax.plot(
                    x,
                    ep[: len(x), d],
                    linewidth=0.9,
                    alpha=0.25,
                    color="black",
                    zorder=1,
                    label="train leader" if d == 0 and i == 0 else None,
                )
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(f"dim {d}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t (prediction time)")
    fig.suptitle(f"{prefix} future action (per-step horizon)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_future_action_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_follower_plots(
    episodes: list[dict[str, np.ndarray]],
    state_norm: Normalizer,
    fig_dir: Path,
    noise_std: float = 0.0,
) -> None:
    import matplotlib.pyplot as plt

    if not episodes:
        return
    states = [np.asarray(ep["state"], dtype=np.float32) for ep in episodes if len(ep.get("state", [])) > 0]
    if not states:
        return
    episodes_norm = [state_norm.normalize(s) for s in states]
    n_dims = episodes_norm[0].shape[1]

    def _plot_series(data_eps: list[np.ndarray], title: str, out_name: str) -> None:
        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        legend_labels = []
        fig, axes = plt.subplots(nrows=n_dims, ncols=1, figsize=(12, 2 * n_dims), sharex=False)
        if n_dims == 1:
            axes = [axes]
        for d, ax in enumerate(axes):
            for i, ep in enumerate(data_eps):
                color = palette[(i // 15) % len(palette)]
                label = None
                if d == 0 and i % 15 == 0:
                    start = i
                    end = min(i + 14, len(data_eps) - 1)
                    label = f"{start}-{end}"
                    legend_labels.append(label)
                ax.plot(ep[:, d], linewidth=0.8, alpha=0.4, color=color, label=label)
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel(f"joint {d}")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("t (per episode)")
        fig.suptitle(title, y=1.02)
        if legend_labels:
            axes[0].legend(title="episode index", ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / out_name, dpi=150)
        plt.close(fig)

    _plot_series(episodes_norm, "Normalized Follower Joint States", "joint_states_normalized.png")
    if noise_std > 0.0:
        rng = np.random.default_rng(0)
        noisy_eps = [
            ep + rng.normal(0.0, noise_std, size=ep.shape).astype(np.float32) for ep in episodes_norm
        ]
        _plot_series(noisy_eps, "Noisy Follower Joint States", "joint_states_noisy_normalized.png")

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

    fig_dir = _ensure_fig_subdir(fig_dir, "online")
    model.eval()
    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    mse_values: list[float] = []
    saved_gif = False
    episode_lengths: list[int] = []

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
            if pred.ndim == 4:
                pred = pred[:, :, 0, :]

            pred_np = pred.detach().cpu().numpy()
            gt_np = action.detach().cpu().numpy()

            # EpisodeDataset で既に正規化済みなので、そのまま比較・描画する
            mse_values.append(float(np.mean((pred_np - gt_np) ** 2)))

            # エピソード単位で保存
            for b in range(pred_np.shape[0]):
                all_pred.append(pred_np[b])
                all_gt.append(gt_np[b])
                episode_lengths.append(int(pred_np[b].shape[0]))

    pred_all = np.concatenate(all_pred, axis=0) if all_pred else np.zeros((0, 1), dtype=np.float32)
    gt_all = np.concatenate(all_gt, axis=0) if all_gt else np.zeros((0, 1), dtype=np.float32)
    mse = float(np.mean(mse_values)) if mse_values else float("nan")

    save_action_figs(
        pred_all=pred_all,
        gt_all=gt_all,
        fig_dir=fig_dir,
        prefix=prefix,
        episode_lengths=episode_lengths if episode_lengths else None,
    )
    return {f"{prefix}_mse": mse, "num_samples": int(len(pred_all))}

# 実機でのテストコード
def run_replay(cfg, model: nn.Module, state_norm: Normalizer, action_norm: Normalizer, device: torch.device, fig_dir: Path):
    from lerobot_utils import Replay  # type: ignore
    from PIL import Image, ImageDraw, ImageFont

    fig_dir = _ensure_fig_subdir(fig_dir, "offline")
    overlay_dir = _ensure_fig_subdir(fig_dir, "overlay")
    future_dir = _ensure_fig_subdir(fig_dir, "future")
    camera_dir = _ensure_fig_subdir(fig_dir, "camera")

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
                replay.send(action=action, fps=1)
            time.sleep(0.2)

    pred_actions: list[np.ndarray] = []
    future_preds: list[np.ndarray] = []
    gt_states: list[np.ndarray] = []
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
    action_horizon = int(getattr(cfg.policy, "action_horizon", 1) or 1)
    chunk_starts: list[int] = []
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
            if action_seq.ndim == 4:
                # 未来予測も保存
                future = action_seq[0, 0].detach().cpu().numpy()  # [K,Da]
                future_preds.append(future)
                action_normed = future[0]
            else:
                action_normed = action_seq[0, 0].detach().cpu().numpy()
        chunk_starts.append(len(pred_actions))
        print(f"Action predicted (normalized): {action_normed}")

        # 正規化された数値を「実際のロボットの角度」に復元（逆正規化）
        action = action_norm.denormalize(action_normed)
        action_t = torch.tensor(action, dtype=torch.float32)

        # グラフ作成用に推論結果を記録（正規化スケールのまま）
        pred_actions.append(action_normed)
        gt_states.append(state_np[0].copy())

        # --- 6. ロボットへの指令送信 ---
        if cfg.replay.send_action:
            # 計算した目標角度を実機のモーターに送信して動かす
            replay.send(
                action=action_t,
                fps=cfg.replay.fps,
            )

        # 可視化用に途中経過を保存（重いので間引く）
        viz_every = int(getattr(cfg.replay, "viz_every", 10) or 10)
        if viz_every > 0 and (step_idx + 1) % viz_every == 0:
            pred_arr = np.asarray(pred_actions, dtype=np.float32)
            gt_arr = np.asarray(gt_states, dtype=np.float32)
            if pred_arr.shape[1] != gt_arr.shape[1]:
                d = min(pred_arr.shape[1], gt_arr.shape[1])
                pred_arr = pred_arr[:, :d]
                gt_arr = gt_arr[:, :d]
            save_action_figs(pred_all=pred_arr, gt_all=gt_arr, fig_dir=fig_dir, prefix="offline_replay")

    # 全ステップ終了後、アクションの推移グラフを保存して終了
    if pred_actions:
        pred_arr = np.asarray(pred_actions, dtype=np.float32)
        gt_arr = np.asarray(gt_states, dtype=np.float32) if gt_states else None
        if gt_arr is not None and pred_arr.shape[1] != gt_arr.shape[1]:
            d = min(pred_arr.shape[1], gt_arr.shape[1])
            pred_arr = pred_arr[:, :d]
            gt_arr = gt_arr[:, :d]
        save_action_figs(pred_all=pred_arr, gt_all=gt_arr, fig_dir=fig_dir, prefix="offline_replay")
        if future_preds:
            leader_traj_eps = None
            try:
                train_eps, test_eps = load_episodes(cfg)
                leader_traj_eps = [
                    action_norm.normalize(np.asarray(ep["action"], dtype=np.float32))
                    for ep in (train_eps + test_eps)
                    if len(ep.get("action", [])) > 0
                ]
            except Exception:
                leader_traj_eps = None
            save_future_action_figs(
                future_preds=future_preds,
                leader_traj_eps=leader_traj_eps,
                fig_dir=future_dir,
                prefix="offline_replay",
            )
        if chunk_starts:
            # 実機フォロワーとの比較
            save_chunked_action_overlay(
                pred_all=pred_arr,
                gt_all=gt_arr,
                chunk_starts=chunk_starts,
                fig_dir=overlay_dir,
                prefix="offline_replay",
                title="offline_replay action (vs real follower)",
                label_gt="real follower",
            )
            # 学習時フォロワーとの比較
            train_eps, test_eps = load_episodes(cfg)
            train_states = [np.asarray(ep["state"], dtype=np.float32) for ep in (train_eps + test_eps)]
            if train_states:
                train_all = np.concatenate(train_states, axis=0).astype(np.float32)
                train_norm = state_norm.normalize(train_all)
                save_chunked_action_overlay(
                    pred_all=pred_arr,
                    gt_all=train_norm,
                    chunk_starts=chunk_starts,
                    fig_dir=overlay_dir,
                    prefix="offline_replay_train",
                    title="offline_replay action (vs train follower)",
                    label_gt="train follower",
                )
                # 右側に joint_states_normalized を並べて1枚にする
                try:
                    from PIL import Image

                    left_path = overlay_dir / "offline_replay_train_action_all_overlay.png"
                    right_path = fig_dir.parent / "follower" / "joint_states_normalized.png"
                    if left_path.exists() and right_path.exists():
                        left = Image.open(left_path).convert("RGB")
                        right = Image.open(right_path).convert("RGB")
                        height = max(left.height, right.height)
                        # 高さを揃える
                        if left.height != height:
                            left = left.resize((int(left.width * height / left.height), height))
                        if right.height != height:
                            right = right.resize((int(right.width * height / right.height), height))
                        merged = Image.new("RGB", (left.width + right.width, height), (255, 255, 255))
                        merged.paste(left, (0, 0))
                        merged.paste(right, (left.width, 0))
                        merged.save(left_path)
                except Exception:
                    pass

    if frames:
        gif_path = camera_dir / "online_camera.gif"
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
