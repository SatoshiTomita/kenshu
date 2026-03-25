from __future__ import annotations

from pathlib import Path
import sys

import hydra
import torch
import numpy as np
try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

_ROOT = Path(__file__).resolve().parent
sys.path.append(str(_ROOT / "src"))

from src.utils.app_helpers import build_model, online_test, run_replay, save_follower_plots
from src.dataloader.episode_dataset import EpisodeDataset, compute_normalizer
from src.dataloader.dataloader import myDataloader
from src.trainer.trainer import Trainer
from src.utils.train_utils import (
    load_episodes,
    load_normalizers,
    load_cfg,
    resolve_device,
    save_cfg,
    save_normalizers,
    set_seed,
    split_indices,
)


def _resolve_model_dir(cfg) -> Path:
    if getattr(cfg.replay, "model_name", ""):
        root = Path(cfg.result_root)
        direct = root / cfg.replay.model_name
        if direct.exists():
            return direct
        # fallback: search nested dirs by name
        matches = [p for p in root.rglob(cfg.replay.model_name) if p.is_dir()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError({"model_dir_ambiguous": [str(p) for p in matches]})
        return direct
    return Path(cfg.result_root) / cfg.train_name


def _load_model(cfg, device: torch.device, state_dim: int, action_dim: int, model_path: Path):
    model = build_model(cfg, state_dim=state_dim, action_dim=action_dim).to(device)
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)

    # 結果保存用のディレクトリを作成
    result_dir = Path(cfg.result_root) / cfg.train_name
    fig_dir = result_dir / "fig"
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # offline_test(推論)を行う処理
    if cfg.mode in ("online_test", "offline_test"):
        model_dir = _resolve_model_dir(cfg)
        fig_dir = model_dir / "fig"
        fig_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = model_dir / "config.yaml"
        if cfg_path.exists():
            loaded_cfg = load_cfg(cfg_path)
        else:
            print({"warn": "model_config_not_found", "path": str(cfg_path)})
            loaded_cfg = cfg
        state_norm, action_norm = load_normalizers(model_dir / "normalizer.yaml")
        model_path = Path(cfg.replay.model_path) if cfg.replay.model_path else model_dir / "best_model.pt"
        model = _load_model(
            loaded_cfg,
            device=device,
            state_dim=len(state_norm.min),
            action_dim=len(action_norm.min),
            model_path=model_path,
        )
        if cfg.replay.enable:
            run_replay(loaded_cfg, model=model, state_norm=state_norm, action_norm=action_norm, device=device, fig_dir=fig_dir)
        print({"mode": str(cfg.mode), "device": str(device)})
        return

    train_episodes, explicit_test_episodes = load_episodes(cfg)

    # test用のエピソードが明示的に指定されている場合は、trainとvalを分割してtestは指定されたものを使用する。
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
    
    # 正規化パラメータの作成
    state_norm, action_norm = compute_normalizer(tr_eps)

    # データセットの作成（学習時は拡張も行う）
    train_ds = EpisodeDataset(
        tr_eps,
        state_norm,
        action_norm,
        augment=bool(cfg.dataset.augment),
        aug_brightness=float(cfg.dataset.aug_brightness),
        aug_contrast=float(cfg.dataset.aug_contrast),
        aug_shift=int(cfg.dataset.aug_shift),
        aug_crop=int(cfg.dataset.aug_crop),
    )
    val_ds = EpisodeDataset(val_eps, state_norm, action_norm)
    test_ds = EpisodeDataset(test_eps, state_norm, action_norm)

    # データがからではないか確認
    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise RuntimeError("Dataset is too small for current split settings.")
    
    
    # lengths = [int(ep["action"].shape[0]) for ep in (tr_eps + val_eps + test_eps)]
    # if len(set(lengths)) > 1:
    #     raise RuntimeError("Episode lengths are not fixed. Use batch_size=1 or add padding.")

    # DatasetからDataloaderを作成＆学習・検証・テスト用に分割
    dataloader = myDataloader(
        batch_size=int(cfg.dataset.batch_size),
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=bool(cfg.dataset.pin_memory),
    )
    train_loader = dataloader.prepare_data(train_ds, shuffle=True)
    val_loader = dataloader.prepare_data(val_ds, shuffle=False)
    test_loader = dataloader.prepare_data(test_ds, shuffle=False)

    # 画像拡張後のサンプルを保存（学習前に1回）
    if bool(cfg.dataset.augment):
        try:
            batch = next(iter(train_loader))
            image = batch[0]  # [B,T,C,H,W]
            if image.ndim == 5:
                # 1エピソード目の先頭フレーム群から16枚
                frames = image[0]  # [T,C,H,W]
            else:
                frames = image
            n = min(4, frames.shape[0])
            h, w = frames.shape[2], frames.shape[3]
            canvas = np.zeros((n * h, 2 * w, 3), dtype=np.uint8)
            # 元画像を同じサンプルから読み直す（拡張前の比較用）
            raw_eps = tr_eps[0]["image"] if tr_eps else frames.detach().cpu().numpy()
            for i in range(n):
                # 左: 元画像
                raw = raw_eps[i]
                if raw.shape[0] == 1:
                    raw = np.repeat(raw, 3, axis=0)
                raw = raw.transpose(1, 2, 0)
                raw = np.clip(raw, 0.0, 1.0) * 255.0
                canvas[i * h : (i + 1) * h, 0:w] = raw.astype(np.uint8)
                # 右: 拡張後
                img = frames[i].detach().cpu().numpy()
                if img.shape[0] == 1:
                    img = np.repeat(img, 3, axis=0)
                img = img.transpose(1, 2, 0)
                img = np.clip(img, 0.0, 1.0) * 255.0
                canvas[i * h : (i + 1) * h, w:2 * w] = img.astype(np.uint8)
            from PIL import Image

            Image.fromarray(canvas).save(fig_dir / "augmented_compare.png")
        except Exception:
            pass

    # モデルの作成
    model = build_model(
        cfg,
        state_dim=len(state_norm.min),
        action_dim=len(action_norm.min),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
    )

    # WandBの初期化
    use_wandb = bool(cfg.wandb.enable) and wandb is not None
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config={"config": cfg.wandb.config, "train_name": cfg.train_name},
        )

    # 学習ループ
    best_val = float("inf")
    best_path = result_dir / "best_model.pt"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        state_noise_std=float(cfg.trainer.state_noise_std),
    )
    for epoch in range(int(cfg.trainer.epochs)):
        epoch_logs = trainer.fit(train_loader, val_loader, epochs=1)[0]
        train_loss = epoch_logs["train_loss"]
        val_loss = epoch_logs["val_loss"]
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        if use_wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
    model.load_state_dict(torch.load(best_path, map_location=device))
    online_metrics = online_test(
        model=model,
        loader=test_loader,
        device=device,
        action_norm=action_norm,
        fig_dir=fig_dir,
        prefix="online",
    )
    save_follower_plots(
        episodes=tr_eps + val_eps + test_eps,
        state_norm=state_norm,
        fig_dir=fig_dir,
        noise_std=float(cfg.trainer.state_noise_std),
    )

    # 再構成画像をグリッドで保存（学習後に1回、4枚を大きめに）
    for image, state, _ in test_loader:
        image = image.to(device)
        state = state.to(device)
        with torch.no_grad():
            _, _, recon = model(image, state)
        recon = recon[0].detach().cpu().numpy()  # [T,C,H,W]
        n = min(4, recon.shape[0])
        grid_h, grid_w = 2, 2
        scale = 2
        c = recon.shape[1]
        h, w = recon.shape[2], recon.shape[3]
        canvas = np.zeros((grid_h * h * scale, grid_w * w * scale, 3), dtype=np.uint8)
        from PIL import Image

        for i in range(n):
            r, cidx = divmod(i, grid_w)
            img = recon[i]
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            img = img.transpose(1, 2, 0)
            img = np.clip(img, 0.0, 1.0) * 255.0
            img_u8 = img.astype(np.uint8)
            img_big = Image.fromarray(img_u8).resize((w * scale, h * scale), resample=Image.NEAREST)
            canvas[
                r * h * scale : (r + 1) * h * scale,
                cidx * w * scale : (cidx + 1) * w * scale,
            ] = np.asarray(img_big)

        Image.fromarray(canvas).save(fig_dir / "recon_grid.png")
        break

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
            "epochs": int(cfg.trainer.epochs),
        }
    )


if __name__ == "__main__":
    main()
