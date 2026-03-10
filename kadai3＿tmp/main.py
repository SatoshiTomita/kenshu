from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import wandb

from src.data.episode_dataset import (
    WindowDataset,
    compute_normalizer,
    get_test_roots,
    get_train_roots,
    load_episodes_from_roots,
    summarize_episode_data,
)
from src.eval.evaluator import run_offline_replay, run_online_test
from src.model.policy import PolicyRNN, VisionPolicyModel
from src.model.vision import VisionCNN
from src.train.trainer import Trainer
from src.utils.io import load_normalizers, resolve_device, save_cfg, save_normalizers, set_seed


def _load_episodes(cfg: DictConfig):
    data_root = Path(cfg.dataset.root)
    train_roots = get_train_roots(
        data_root=data_root,
        train_dir=cfg.dataset.train_dir,
        left_dirname=cfg.dataset.left_dirname,
        right_dirname=cfg.dataset.right_dirname,
        variant=cfg.dataset.variant,
    )
    train_episodes = load_episodes_from_roots(
        roots=train_roots,
        image_key=cfg.dataset.image_key,
        state_key=cfg.dataset.state_key,
        action_key=cfg.dataset.action_key,
        action_states_filename=cfg.dataset.action_states_filename,
        image_states_filename=cfg.dataset.image_states_filename,
        joint_states_filename=cfg.dataset.joint_states_filename,
    )
    if len(train_episodes) == 0:
        raise RuntimeError("No train episodes found. Check dataset.root/train and left/right folders.")
    test_episodes = load_episodes_from_roots(
        roots=get_test_roots(data_root=data_root, test_dir=cfg.dataset.test_dir),
        image_key=cfg.dataset.image_key,
        state_key=cfg.dataset.state_key,
        action_key=cfg.dataset.action_key,
        action_states_filename=cfg.dataset.action_states_filename,
        image_states_filename=cfg.dataset.image_states_filename,
        joint_states_filename=cfg.dataset.joint_states_filename,
    )

    return train_episodes, test_episodes


def _build_model(cfg: DictConfig, state_dim: int, action_dim: int):
    vision = VisionCNN(
        in_channels=cfg.vision.in_channels,
        conv_channels=list(cfg.vision.conv_channels),
        kernels=list(cfg.vision.kernels),
        strides=list(cfg.vision.strides),
        paddings=list(cfg.vision.paddings),
        feature_dim=cfg.vision.feature_dim,
    )
    policy = PolicyRNN(
        input_dim=cfg.vision.feature_dim + state_dim,
        hidden_dim=cfg.policy.hidden_dim,
        output_dim=action_dim,
        num_layers=cfg.policy.num_layers,
        rnn_type=cfg.policy.rnn_type,
    )
    return VisionPolicyModel(vision=vision, policy=policy)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
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

        replay_metrics = run_offline_replay(
            cfg=cfg,
            model=model,
            state_norm=state_norm,
            action_norm=action_norm,
            device=device,
            seq_len=int(cfg.dataset.seq_len),
        )
        print({"mode": "offline_test", **replay_metrics})
        return

    train_episodes, explicit_test_episodes = _load_episodes(cfg)
    print({"train_data_summary": summarize_episode_data(train_episodes)})
    if len(explicit_test_episodes) > 0:
        print({"test_data_summary": summarize_episode_data(explicit_test_episodes)})

    tr_eps, val_eps = train_test_split(
        train_episodes,
        test_size=float(cfg.dataset.val_ratio),
        random_state=int(cfg.seed),
        shuffle=True,
    )

    if len(explicit_test_episodes) > 0:
        test_eps = explicit_test_episodes
    else:
        tr_eps, test_eps = train_test_split(
            tr_eps,
            test_size=float(cfg.dataset.test_ratio),
            random_state=int(cfg.seed),
            shuffle=True,
        )

    state_norm, action_norm = compute_normalizer(tr_eps)

    train_ds = WindowDataset(tr_eps, int(cfg.dataset.seq_len), state_norm, action_norm)
    val_ds = WindowDataset(val_eps, int(cfg.dataset.seq_len), state_norm, action_norm)
    test_ds = WindowDataset(test_eps, int(cfg.dataset.seq_len), state_norm, action_norm)

    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise RuntimeError("Dataset is too small for current seq_len/split settings.")

    dl_kwargs = dict(
        batch_size=int(cfg.dataset.batch_size),
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=bool(cfg.dataset.pin_memory),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kwargs)

    sample_image, sample_state, sample_action = train_ds[0]
    model = _build_model(
        cfg,
        state_dim=int(sample_state.shape[-1]),
        action_dim=int(sample_action.shape[-1]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
    )

    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config={"config": cfg.wandb.config, "train_name": cfg.train_name},
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        save_dir=result_dir,
        use_wandb=bool(cfg.wandb.enable),
        grad_clip_norm=float(cfg.trainer.grad_clip_norm),
    )
    train_log = trainer.train(train_loader, val_loader, epochs=int(cfg.trainer.epochs))

    # Reload best model for evaluation.
    model.load_state_dict(torch.load(result_dir / "best_model.pt", map_location=device))
    online_metrics = run_online_test(
        model=model,
        loader=test_loader,
        device=device,
        action_norm=action_norm,
        fig_dir=fig_dir,
    )

    save_cfg(cfg, result_dir / "config.yaml")
    save_normalizers(state_norm, action_norm, result_dir / "normalizer.yaml")

    if cfg.wandb.enable:
        wandb.log({"final_online_mse": online_metrics["online_mse"]})
        wandb.finish()

    print(
        {
            "mode": "train",
            "device": str(device),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "best_model": str(result_dir / "best_model.pt"),
            "online_mse": online_metrics["online_mse"],
            "epochs": len(train_log),
        }
    )

    if cfg.replay.enable and cfg.replay.use_after_train:
        replay_metrics = run_offline_replay(
            cfg=cfg,
            model=model,
            state_norm=state_norm,
            action_norm=action_norm,
            device=device,
            seq_len=int(cfg.dataset.seq_len),
        )
        print({"mode": "replay_after_train", **replay_metrics})


if __name__ == "__main__":
    main()
