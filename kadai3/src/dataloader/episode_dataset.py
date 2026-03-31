from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import blosc2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


_ID_PATTERN = re.compile(r"_(\d+_\d+)$")


@dataclass
class Normalizer:
    min: np.ndarray
    max: np.ndarray
    eps: float = 1.0e-8

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        denom = np.maximum(self.max - self.min, self.eps)
        return 2.0 * (x - self.min) / denom - 1.0

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        scale = np.maximum(self.max - self.min, self.eps)
        return (x + 1.0) * 0.5 * scale + self.min

    def to_dict(self) -> dict:
        return {
            "min": self.min.astype(float).tolist(),
            "max": self.max.astype(float).tolist(),
            "eps": float(self.eps),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Normalizer":
        return cls(
            min=np.asarray(data["min"], dtype=np.float32),
            max=np.asarray(data["max"], dtype=np.float32),
            eps=float(data.get("eps", 1.0e-8)),
        )


def _sort_key_by_last_number(path: Path) -> tuple[int, str]:
    stem = path.stem
    m = re.search(r"_(\d+)$", stem)
    if m:
        return int(m.group(1)), stem
    return 10**9, stem


def _episode_id_from_stem(stem: str) -> str:
    m = _ID_PATTERN.search(stem)
    if not m:
        raise ValueError(f"Cannot parse episode id from stem: {stem}")
    return m.group(1)


def _load_blosc2_tensor(path: Path) -> np.ndarray:
    schunk = blosc2.open(str(path))
    _, shape, dtype = schunk.vlmeta[b"__pack_tensor__"]
    return np.frombuffer(schunk[:], dtype=np.dtype(dtype)).reshape(shape)


def _ensure_image_seq(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim != 4:
        raise ValueError(f"Expected image with shape [T,C,H,W] or [T,H,W,C], got {img.shape}")
    if img.shape[1] in (1, 3):
        out = img
    elif img.shape[-1] in (1, 3):
        out = np.transpose(img, (0, 3, 1, 2))
    else:
        raise ValueError(f"Unexpected image channel layout: {img.shape}")
    out = out.astype(np.float32)
    if out.max() > 1.0:
        out = out / 255.0
    return out


def _resolve_concat_file(base: Path, filename: str) -> Path | None:
    primary = base / filename
    if primary.exists():
        return primary
    if filename.endswith("states.blosc2"):
        alt = base / filename.replace("states.blosc2", "state.blosc2")
        if alt.exists():
            return alt
    return None


def _split_concat(arr: np.ndarray) -> list[np.ndarray]:
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [np.asarray(x) for x in arr.tolist()]
    if arr.ndim >= 3:
        return [np.asarray(x) for x in arr]
    return [np.asarray(arr)]


def _load_concat_episodes(
    root: Path,
    image_key: str,
    state_key: str,
    action_key: str,
    action_states_filename: str,
    image_states_filename: str,
    joint_states_filename: str,
) -> list[dict[str, np.ndarray]]:
    action_path = _resolve_concat_file(root, action_states_filename)
    image_path = _resolve_concat_file(root, image_states_filename)
    state_path = _resolve_concat_file(root, joint_states_filename)
    if action_path is None or image_path is None or state_path is None:
        return []
    action_concat = _load_blosc2_tensor(action_path)
    image_concat = _load_blosc2_tensor(image_path)
    state_concat = _load_blosc2_tensor(state_path)
    action_eps = _split_concat(action_concat)
    image_eps = _split_concat(image_concat)
    state_eps = _split_concat(state_concat)
    n = min(len(action_eps), len(image_eps), len(state_eps))
    episodes: list[dict[str, np.ndarray]] = []
    for i in range(n):
        image = _ensure_image_seq(image_eps[i])
        state = np.asarray(state_eps[i], dtype=np.float32)
        action = np.asarray(action_eps[i], dtype=np.float32)
        t = min(len(image), len(state), len(action))
        if t <= 0:
            continue
        episodes.append(
            {
                image_key: image[:t],
                state_key: state[:t],
                action_key: action[:t],
            }
        )
    return episodes


def _load_episode_files(
    root: Path,
    image_key: str,
    state_key: str,
    action_key: str,
) -> list[dict[str, np.ndarray]]:
    episodes: list[dict[str, np.ndarray]] = []
    for path in sorted(root.glob("*.npz"), key=_sort_key_by_last_number):
        data = np.load(path)
        if image_key not in data or state_key not in data or action_key not in data:
            continue
        image = _ensure_image_seq(data[image_key])
        state = np.asarray(data[state_key], dtype=np.float32)
        action = np.asarray(data[action_key], dtype=np.float32)
        t = min(len(image), len(state), len(action))
        if t <= 0:
            continue
        episodes.append(
            {
                image_key: image[:t],
                state_key: state[:t],
                action_key: action[:t],
            }
        )
    for path in sorted(root.glob("*.pt")) + sorted(root.glob("*.pth")):
        data = torch.load(path, map_location="cpu")
        if not isinstance(data, dict):
            continue
        if image_key not in data or state_key not in data or action_key not in data:
            continue
        image = _ensure_image_seq(data[image_key])
        state = np.asarray(data[state_key], dtype=np.float32)
        action = np.asarray(data[action_key], dtype=np.float32)
        t = min(len(image), len(state), len(action))
        if t <= 0:
            continue
        episodes.append(
            {
                image_key: image[:t],
                state_key: state[:t],
                action_key: action[:t],
            }
        )
    return episodes


def load_episodes_from_roots(
    roots: list[Path],
    image_key: str,
    state_key: str,
    action_key: str,
    action_states_filename: str,
    image_states_filename: str,
    joint_states_filename: str,
) -> list[dict[str, np.ndarray]]:
    episodes: list[dict[str, np.ndarray]] = []
    for root in roots:
        if not root.exists():
            continue
        eps = _load_concat_episodes(
            root=root,
            image_key=image_key,
            state_key=state_key,
            action_key=action_key,
            action_states_filename=action_states_filename,
            image_states_filename=image_states_filename,
            joint_states_filename=joint_states_filename,
        )
        if not eps:
            eps = _load_episode_files(root, image_key=image_key, state_key=state_key, action_key=action_key)
        for ep in eps:
            image = ep[image_key]
            state = ep[state_key]
            action = ep[action_key]
            t = min(len(image), len(state), len(action))
            if t <= 0:
                continue
            episodes.append(
                {
                    "image": image[:t],
                    "state": state[:t],
                    "action": action[:t],
                }
            )
    return episodes


def get_train_roots(
    data_root: Path,
    train_dir: str,
    left_dirname: str,
    right_dirname: str,
    variant: str,
) -> list[Path]:
    base = data_root / train_dir
    if variant == "both":
        return [p for p in [base / left_dirname, base / right_dirname] if p.exists()]
    if variant == "left_only":
        return [p for p in [base / left_dirname] if p.exists()]
    return [p for p in [base / right_dirname] if p.exists()]


def get_test_roots(data_root: Path, test_dir: str) -> list[Path]:
    base = data_root / test_dir
    if (base / "left").exists() or (base / "right").exists():
        roots = []
        if (base / "left").exists():
            roots.append(base / "left")
        if (base / "right").exists():
            roots.append(base / "right")
        return roots
    return [base]


def compute_normalizer(episodes: list[dict[str, np.ndarray]]) -> tuple[Normalizer, Normalizer]:
    if not episodes:
        raise ValueError("episodes is empty")
    state_all = np.concatenate([ep["state"] for ep in episodes], axis=0).astype(np.float32)
    action_all = np.concatenate([ep["action"] for ep in episodes], axis=0).astype(np.float32)
    state_norm = Normalizer(min=state_all.min(axis=0), max=state_all.max(axis=0))
    action_norm = Normalizer(min=action_all.min(axis=0), max=action_all.max(axis=0))
    return state_norm, action_norm


class EpisodeDataset(Dataset):
    def __init__(
        self,
        episodes: list[dict[str, np.ndarray]],
        state_norm: Normalizer,
        action_norm: Normalizer,
        augment: bool = False,
        aug_brightness: float = 0.0,
        aug_contrast: float = 0.0,
        aug_shift: int = 0,
        aug_crop: int = 0,
    ):
        self.episodes = episodes
        self.state_norm = state_norm
        self.action_norm = action_norm
        self.augment = bool(augment)
        self.aug_brightness = float(aug_brightness)
        self.aug_contrast = float(aug_contrast)
        self.aug_shift = int(aug_shift)
        self.aug_crop = int(aug_crop)
        print({"episodes": len(episodes), "mode": "episode"})

    def __len__(self) -> int:
        return len(self.episodes)

    def _shift(self, img: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        # img: [T,C,H,W]
        if dx == 0 and dy == 0:
            return img
        t, c, h, w = img.shape
        out = torch.zeros_like(img)
        x_from = max(-dx, 0)
        x_to = min(w - dx, w)
        y_from = max(-dy, 0)
        y_to = min(h - dy, h)
        out[:, :, y_from + dy : y_to + dy, x_from + dx : x_to + dx] = img[:, :, y_from:y_to, x_from:x_to]
        return out

    def __getitem__(self, idx: int):
        ep = self.episodes[idx]
        image = ep["image"].astype(np.float32)  # [T,C,H,W]
        if self.augment:
            img_t = torch.from_numpy(image)
            if self.aug_brightness > 0:
                delta = (torch.rand(1) * 2 - 1) * self.aug_brightness
                img_t = img_t + delta
            if self.aug_contrast > 0:
                factor = 1.0 + (torch.rand(1) * 2 - 1) * self.aug_contrast
                mean = img_t.mean(dim=(-2, -1), keepdim=True)
                img_t = (img_t - mean) * factor + mean
            if self.aug_shift > 0:
                dx = int(torch.randint(-self.aug_shift, self.aug_shift + 1, (1,)).item())
                dy = int(torch.randint(-self.aug_shift, self.aug_shift + 1, (1,)).item())
                img_t = self._shift(img_t, dx, dy)
            if self.aug_crop > 0:
                t, c, h, w = img_t.shape
                crop = min(self.aug_crop, h - 1, w - 1)
                top = int(torch.randint(0, crop + 1, (1,)).item())
                left = int(torch.randint(0, crop + 1, (1,)).item())
                bottom = h - (crop - top)
                right = w - (crop - left)
                img_t = img_t[:, :, top:bottom, left:right]
                img_t = F.interpolate(img_t, size=(h, w), mode="bilinear", align_corners=False)
            img_t = torch.clamp(img_t, 0.0, 1.0)
            image = img_t.numpy()
        state = self.state_norm.normalize(ep["state"]).astype(np.float32)
        action = self.action_norm.normalize(ep["action"]).astype(np.float32)
        k = 20  # 追加ステップ数
      image = np.concatenate([image, np.repeat(image[59:60], k, axis=0)], axis=0)
      state = np.concatenate([state, np.repeat(state[59:60], k, axis=0)], axis=0)
      action = np.concatenate([action, np.repeat(action[59:60], k, axis=0)], axis=0)
        return image, state, action
