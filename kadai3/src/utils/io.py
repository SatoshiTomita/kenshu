from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.episode_dataset import Normalizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def save_cfg(cfg, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def save_normalizers(state_norm: Normalizer, action_norm: Normalizer, path: Path):
    payload = {"state": state_norm.to_dict(), "action": action_norm.to_dict()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def load_normalizers(path: Path) -> tuple[Normalizer, Normalizer]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Normalizer.from_dict(data["state"]), Normalizer.from_dict(data["action"])
