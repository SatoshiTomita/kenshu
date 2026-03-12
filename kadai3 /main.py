from __future__ import annotations

from pathlib import Path
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

from src.data.episode_dataset import (
    Normalizer,
    WindowDataset,
    compute_normalizer,
    get_test_roots,
    get_train_roots,
    load_episodes_from_roots,
    summarize_episode_data,
)
from src.models.policy import PolicyNetwork
from src.models.vision import VisionNetwork


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_cfg: str) -> torch.device:
    d = device_cfg.lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _save_cfg(cfg: DictConfig, path: Path) -> None:
    data = OmegaConf.to_container(cfg, resolve=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def _save_normalizers(state_norm: Normalizer, action_norm: Normalizer, path: Path) -> None:
    payload = {"state": state_norm.to_dict(), "action": action_norm.to_dict()}
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _load_normalizers(path: Path) -> tuple[Normalizer, Normalizer]:
    data = yaml.safe_load(path.read_text())
    return Normalizer.from_dict(data["state"]), Normalizer.from_dict(data["action"])


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


def _split_indices(n: int, val_ratio: float, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    n_train = max(n - n_val - n_test, 1)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val : n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


class VisionPolicyModel(nn.Module):
    def __init__(self, vision: VisionNetwork, policy: PolicyNetwork):
        super().__init__()
        self.vision = vision
        self.policy = policy

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        feature = self.vision(image)
        return self.policy(feature, state)


def _build_model(cfg: DictConfig, state_dim: int, action_dim: int) -> nn.Module:
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


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = nn.MSELoss()
    total = 0.0
    count = 0
    for image, state, action in loader:
        image = image.to(device)
        state = state.to(device)
        action = action.to(device)
        pred = model(image, state)
        loss = loss_fn(pred, action)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss.item())
        count += 1
    return total / max(count, 1)


def _online_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_norm: Normalizer,
    fig_dir: Path,
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    _set_seed(int(cfg.seed))
    device = _resolve_device(cfg.device)

    result_dir = Path(cfg.result_root) / cfg.train_name
    fig_dir = result_dir / "fig"
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if cfg.mode == "offline_test":
        norm_path = result_dir / "normalizer.yaml"
        state_norm, action_norm = _load_normalizers(norm_path)
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

            image_q: list[torch.Tensor] = []
            state_q: list[np.ndarray] = []
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
                if cfg.replay.send_action:
                    replay.send(
                        action=action_t,
                        fps=cfg.replay.fps,
                        split=cfg.replay.split,
                        ema=cfg.replay.ema,
                    )
        print({"mode": "offline_test", "device": str(device)})
        return

    train_episodes, explicit_test_episodes = _load_episodes(cfg)
    print({"train_data_summary": summarize_episode_data(train_episodes)})
    if len(explicit_test_episodes) > 0:
        print({"test_data_summary": summarize_episode_data(explicit_test_episodes)})

    if len(explicit_test_episodes) > 0:
        train_idx, val_idx, _ = _split_indices(
            len(train_episodes), float(cfg.dataset.val_ratio), 0.0, int(cfg.seed)
        )
        tr_eps = [train_episodes[i] for i in train_idx]
        val_eps = [train_episodes[i] for i in val_idx] if len(val_idx) > 0 else tr_eps
        test_eps = explicit_test_episodes
    else:
        train_idx, val_idx, test_idx = _split_indices(
            len(train_episodes),
            float(cfg.dataset.val_ratio),
            float(cfg.dataset.test_ratio),
            int(cfg.seed),
        )
        tr_eps = [train_episodes[i] for i in train_idx]
        val_eps = [train_episodes[i] for i in val_idx] if len(val_idx) > 0 else tr_eps
        test_eps = [train_episodes[i] for i in test_idx] if len(test_idx) > 0 else val_eps

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

    use_wandb = bool(cfg.wandb.enable) and wandb is not None
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config={"config": cfg.wandb.config, "train_name": cfg.train_name},
        )

    best_val = float("inf")
    best_path = result_dir / "best_model.pt"
    history: list[dict[str, float]] = []
    for epoch in range(int(cfg.trainer.epochs)):
        train_loss = _run_epoch(model, train_loader, device, optimizer)
        val_loss = _run_epoch(model, val_loader, device, None)
        history.append({"train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        if use_wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

    # reload best
    model.load_state_dict(torch.load(best_path, map_location=device))
    online_metrics = _online_test(
        model=model,
        loader=test_loader,
        device=device,
        action_norm=action_norm,
        fig_dir=fig_dir,
    )

    _save_cfg(cfg, result_dir / "config.yaml")
    _save_normalizers(state_norm, action_norm, result_dir / "normalizer.yaml")

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
