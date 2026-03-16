from __future__ import annotations

from pathlib import Path
import sys

import hydra
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

_ROOT = Path(__file__).resolve().parent
sys.path.append(str(_ROOT / "src"))

from config_schema import MainConfig
from src.data.episode_dataset import (
    Normalizer,
    WindowDataset,
    compute_normalizer,
    summarize_episode_data,
)
from src.dataloader.dataloader import myDataloader
from src.models.policy import PolicyNetwork
from src.models.vision import VisionNetwork
from src.models.vision_policy import VisionPolicyModel
from src.trainer.trainer import Trainer
from src.utils.train_utils import (
    load_episodes,
    load_normalizers,
    resolve_device,
    save_cfg,
    save_normalizers,
    set_seed,
    split_indices,
)


# Vision + Policy を組み合わせたモデルを作成
def _build_model(cfg: MainConfig, state_dim: int, action_dim: int) -> nn.Module:
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


# テストデータで推論してMSEと可視化を保存
def _online_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_norm: Normalizer,
    fig_dir: Path,
    use_state: bool,
) -> dict:
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    mse_values: list[float] = []

    with torch.no_grad():
        for image, state, action in loader:
            image = image.to(device)
            state = state.to(device)
            if not use_state:
                state = torch.zeros_like(state)
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

    def _moving_avg(x: np.ndarray, win: int = 5) -> np.ndarray:
        # 簡易な移動平均（端は端値でパディング）
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

    pred_s = _moving_avg(pred_all, win=5)
    gt_s = _moving_avg(gt_all, win=5)
    n_dims = min(pred_s.shape[1], gt_s.shape[1])

    # 次元ごとの図（平滑化済み）
    for d in range(n_dims):
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.plot(gt_s[:, d], label="gt", linewidth=1.0)
        ax.plot(pred_s[:, d], label="pred", linewidth=1.0)
        ax.set_title(f"Online test action dim {d} (smoothed)")
        ax.legend()
        fig.savefig(fig_dir / f"online_action_dim_{d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 全次元を1枚にまとめた図（平滑化済み）
    if n_dims > 0:
        ncols = 2
        nrows = (n_dims + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows), squeeze=False)
        for d in range(n_dims):
            r, c = divmod(d, ncols)
            ax = axes[r][c]
            ax.plot(gt_s[:, d], label="follower", linewidth=1.0)
            ax.plot(pred_s[:, d], label="pred", linewidth=1.0)
            ax.set_title(f"Action dim {d} (smoothed)")
            ax.legend()
        for idx in range(n_dims, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / "online_action_all.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {"online_mse": mse, "num_samples": int(len(pred_all))}


# 学習/評価のエントリポイント
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: MainConfig):
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)

    result_dir = Path(cfg.result_root) / cfg.train_name
    fig_dir = result_dir / "fig"
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if cfg.mode == "offline_test":
        norm_path = result_dir / "normalizer.yaml"
        state_norm, action_norm = load_normalizers(norm_path)
        state_dim = len(state_norm.min)
        action_dim = len(action_norm.min)
        model = _build_model(cfg, state_dim=state_dim, action_dim=action_dim).to(device)
        model_path = Path(cfg.replay.model_path) if cfg.replay.model_path else result_dir / "best_model.pt"
        if not model_path.exists():
            raise RuntimeError(f"Model not found: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        # Offline replay requires lerobot_utils; only run if enabled.
        if cfg.replay.enable:
            from lerobot_utils import Replay  # type: ignore

            replay = Replay(
                height=cfg.replay.height,
                width=cfg.replay.width,
                camera_id=tuple(cfg.replay.camera_id),
                is_higher_port=cfg.replay.is_higher_port,
                leader_port=cfg.replay.leader_port,
                follower_port=cfg.replay.follower_port,
                calibration_name=cfg.replay.calibration_name,
            )

            image_q: list[torch.Tensor] = []
            state_q: list[np.ndarray] = []
            model.eval()
            for step in range(int(cfg.replay.steps)):
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
                # デバッグ: キューのサイズを確認
                print(f"Step: {step}, Queue size: {len(image_q)}")
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
                # デバッグ: 予測アクションを確認
                print(f"Action predicted (normalized): {action_normed}")
                action = action_norm.denormalize(action_normed)
                action_t = torch.tensor(action, dtype=torch.float32)
                if cfg.replay.send_action:
                    replay.send(
                        action=action_t,
                        fps=cfg.replay.fps,
                        split=cfg.replay.split,
                        ema=cfg.replay.ema,
                    )
        print({"mode": "offline_test", "device": str(device)})
        return

    train_episodes, explicit_test_episodes = load_episodes(cfg)
    # 学習に使うエピソードの概要を表示（件数・次元などの確認用）
    print({"train_data_summary": summarize_episode_data(train_episodes)})
    if len(explicit_test_episodes) > 0:
        print({"test_data_summary": summarize_episode_data(explicit_test_episodes)})

    if len(explicit_test_episodes) > 0:
        train_idx, val_idx, _ = split_indices(
            len(train_episodes), float(cfg.dataset.val_ratio), 0.0, int(cfg.seed)
        )
        tr_eps = [train_episodes[i] for i in train_idx]
        val_eps = [train_episodes[i] for i in val_idx] if len(val_idx) > 0 else tr_eps
        test_eps = explicit_test_episodes
    else:
        train_idx, val_idx, test_idx = split_indices(
            len(train_episodes),
            float(cfg.dataset.val_ratio),
            float(cfg.dataset.test_ratio),
            int(cfg.seed),
        )
        tr_eps = [train_episodes[i] for i in train_idx]
        val_eps = [train_episodes[i] for i in val_idx] if len(val_idx) > 0 else tr_eps
        test_eps = [train_episodes[i] for i in test_idx] if len(test_idx) > 0 else val_eps

    # 学習用データだけから正規化パラメータを計算（テスト漏洩を防ぐ）
    state_norm, action_norm = compute_normalizer(tr_eps)
    # 時系列をseq_lenの窓に切り出したDatasetを作成（__getitem__内で正規化）
    train_ds = WindowDataset(
        tr_eps,
        int(cfg.dataset.seq_len),
        state_norm,
        action_norm,
        window_stride=int(cfg.dataset.window_stride),
        augment=bool(cfg.dataset.augment),
        aug_brightness=float(cfg.dataset.aug_brightness),
        aug_contrast=float(cfg.dataset.aug_contrast),
        aug_shift=int(cfg.dataset.aug_shift),
    )
    val_ds = WindowDataset(
        val_eps,
        int(cfg.dataset.seq_len),
        state_norm,
        action_norm,
        window_stride=int(cfg.dataset.window_stride),
    )
    test_ds = WindowDataset(
        test_eps,
        int(cfg.dataset.seq_len),
        state_norm,
        action_norm,
        window_stride=int(cfg.dataset.window_stride),
    )

    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise RuntimeError("Dataset is too small for current seq_len/split settings.")

    # ミニバッチ生成用のDataLoaderを用意
    dataloader = myDataloader(
        batch_size=int(cfg.dataset.batch_size),
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=bool(cfg.dataset.pin_memory),
    )
    train_loader = dataloader.prepare_data(train_ds, shuffle=True)
    val_loader = dataloader.prepare_data(val_ds, shuffle=False)
    test_loader = dataloader.prepare_data(test_ds, shuffle=False)

    # サンプルからstate/action次元を推定してモデルを構築
    sample_image, sample_state, sample_action = train_ds[0]
    model = _build_model(
        cfg,
        state_dim=int(sample_state.shape[-1]),
        action_dim=int(sample_action.shape[-1]),
    ).to(device)

    # CNN+RNNの学習用オプティマイザ
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
    )

    use_wandb = bool(cfg.wandb.enable) and wandb is not None
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config={"config": cfg.wandb.config, "train_name": cfg.train_name},
        )

    # 学習と検証（val lossが最小のモデルを保存）
    best_val = float("inf")
    best_path = result_dir / "best_model.pt"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        use_state=bool(cfg.trainer.use_state),
        state_noise_std=float(cfg.trainer.state_noise_std),
    )
    history: list[dict[str, float]] = []
    for epoch in range(int(cfg.trainer.epochs)):
        epoch_logs = trainer.fit(train_loader, val_loader, epochs=1)[0]
        train_loss = epoch_logs["train_loss"]
        val_loss = epoch_logs["val_loss"]
        history.append({"train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        if use_wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

    # ベストモデルでテストデータを評価（図も保存）
    model.load_state_dict(torch.load(best_path, map_location=device))
    online_metrics = _online_test(
        model=model,
        loader=test_loader,
        device=device,
        action_norm=action_norm,
        fig_dir=fig_dir,
        use_state=bool(cfg.trainer.use_state),
    )

    # 再利用用に設定と正規化パラメータを保存
    save_cfg(cfg, result_dir / "config.yaml")
    save_normalizers(state_norm, action_norm, result_dir / "normalizer.yaml")

    if use_wandb:
        wandb.log({"final_online_mse": online_metrics["online_mse"]})
        wandb.finish()

    print(
        {
            "mode": "train",
            "device": str(device),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "best_model": str(best_path),
            "online_mse": online_metrics["online_mse"],
            "epochs": len(history),
        }
    )


if __name__ == "__main__":
    main()
