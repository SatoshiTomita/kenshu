from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import torch
import yaml

from conf import MainConfig
from data.episode_dataset import Normalizer, get_test_roots, get_train_roots, load_episodes_from_roots


def add_src_to_sys_path(root: Path) -> None:
    # src/ をモジュール検索パスに追加（実行位置に依存せずimportできるようにする）
    sys.path.append(str(root / "src"))


def set_seed(seed: int) -> None:
    # 乱数シード固定（再現性確保）
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> torch.device:
    # デバイス設定を解決（cpu/cuda/mps/auto）
    d = device_cfg.lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_cfg(cfg: MainConfig, path: Path) -> None:
    # Hydra設定をYAMLで保存
    data = OmegaConf.to_container(cfg, resolve=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def save_normalizers(state_norm: Normalizer, action_norm: Normalizer, path: Path) -> None:
    # 正規化パラメータを保存
    payload = {"state": state_norm.to_dict(), "action": action_norm.to_dict()}
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def load_normalizers(path: Path) -> tuple[Normalizer, Normalizer]:
    # 正規化パラメータを読み込み
    data = yaml.safe_load(path.read_text())
    return Normalizer.from_dict(data["state"]), Normalizer.from_dict(data["action"])


def load_episodes(cfg: MainConfig):
    # データルートとvariant設定に応じてエピソードを読み込む
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


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int):
    # train/val/test の分割インデックスを作成
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
