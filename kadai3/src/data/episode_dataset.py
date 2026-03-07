from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import blosc2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Episode:
    image: np.ndarray
    state: np.ndarray
    action: np.ndarray
    source_file: str


class Normalizer:
    """Per-dimension min-max normalization to [-1, 1]."""

    def __init__(self, min_value: np.ndarray, max_value: np.ndarray):
        self.min = min_value.astype(np.float32)
        self.max = max_value.astype(np.float32)
        self.range = np.clip((self.max - self.min).astype(np.float32), 1.0e-6, None)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        return 2.0 * ((x - self.min) / self.range) - 1.0

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        return ((x + 1.0) * 0.5) * self.range + self.min

    def to_dict(self) -> dict:
        return {"method": "minmax_-1_1", "min": self.min.tolist(), "max": self.max.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "Normalizer":
        return cls(
            np.asarray(data["min"], dtype=np.float32),
            np.asarray(data["max"], dtype=np.float32),
        )


def _to_chw(image: np.ndarray) -> np.ndarray:
    # Input can be [T,H,W,C], [T,C,H,W], [H,W,C], [C,H,W]
    if image.ndim == 3:
        image = image[None, ...]
    if image.ndim != 4:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.shape[-1] in (1, 3):
        image = np.transpose(image, (0, 3, 1, 2))
    return image.astype(np.float32)


def _as_time_series(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x[None, ...]
    if x.ndim != 2:
        raise ValueError(f"Expected [T,D], but got {x.shape}")
    return x.astype(np.float32)


def _load_dense_file(path: Path, image_key: str, state_key: str, action_key: str) -> Episode:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        image = data[image_key]
        state = data[state_key]
        action = data[action_key]
    elif suffix in (".pt", ".pth"):
        data = torch.load(path, map_location="cpu")
        image = data[image_key]
        state = data[state_key]
        action = data[action_key]
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    if torch.is_tensor(state):
        state = state.detach().cpu().numpy()
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()

    image = _to_chw(np.asarray(image))
    state = _as_time_series(np.asarray(state))
    action = _as_time_series(np.asarray(action))

    t = min(len(image), len(state), len(action))
    if t < 1:
        raise ValueError(f"Episode is empty: {path}")

    return Episode(image=image[:t], state=state[:t], action=action[:t], source_file=str(path))


def _find_concat_data_dirs(
    root: Path,
    action_filename: str,
    image_filename: str,
    state_filename: str,
) -> list[Path]:
    dirs: list[Path] = []
    # Directory itself can be the concatenated data dir.
    if (
        (root / action_filename).exists()
        and (root / image_filename).exists()
        and (root / state_filename).exists()
    ):
        dirs.append(root)

    # Or a nested directory under root.
    for p in root.rglob(action_filename):
        d = p.parent
        if (d / image_filename).exists() and (d / state_filename).exists():
            dirs.append(d)

    unique = sorted({d.resolve() for d in dirs})
    return [Path(u) for u in unique]


def _load_concat_episodes_from_dir(
    concat_dir: Path,
    action_filename: str,
    image_filename: str,
    state_filename: str,
) -> list[Episode]:
    action_arr = np.asarray(blosc2.load_array(str(concat_dir / action_filename)), dtype=np.float32)
    image_arr = np.asarray(blosc2.load_array(str(concat_dir / image_filename)), dtype=np.float32)
    state_arr = np.asarray(blosc2.load_array(str(concat_dir / state_filename)), dtype=np.float32)

    if action_arr.ndim != 3:
        raise ValueError(f"{action_filename} must be [N,T,Da], got {action_arr.shape}")
    if state_arr.ndim != 3:
        raise ValueError(f"{state_filename} must be [N,T,Dq], got {state_arr.shape}")
    if image_arr.ndim != 5:
        raise ValueError(f"{image_filename} must be [N,T,C,H,W] or [N,T,H,W,C], got {image_arr.shape}")

    n = min(action_arr.shape[0], image_arr.shape[0], state_arr.shape[0])
    episodes: list[Episode] = []
    for i in range(n):
        image = _to_chw(np.asarray(image_arr[i]))
        state = _as_time_series(np.asarray(state_arr[i]))
        action = _as_time_series(np.asarray(action_arr[i]))
        t = min(len(image), len(state), len(action))
        if t < 1:
            continue
        episodes.append(
            Episode(
                image=image[:t],
                state=state[:t],
                action=action[:t],
                source_file=f"{concat_dir}#{i}",
            )
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
) -> list[Episode]:
    episodes: list[Episode] = []

    for root in roots:
        if not root.exists():
            continue

        # 1) Prefer concatenated blosc2 data directories if they exist.
        concat_dirs = _find_concat_data_dirs(
            root,
            action_filename=action_states_filename,
            image_filename=image_states_filename,
            state_filename=joint_states_filename,
        )
        for concat_dir in concat_dirs:
            episodes.extend(
                _load_concat_episodes_from_dir(
                    concat_dir,
                    action_filename=action_states_filename,
                    image_filename=image_states_filename,
                    state_filename=joint_states_filename,
                )
            )

        # 2) Fallback: episode-per-file format.
        files: list[Path] = []
        files.extend(sorted(root.rglob("*.npz")))
        files.extend(sorted(root.rglob("*.pt")))
        files.extend(sorted(root.rglob("*.pth")))
        for f in files:
            episodes.append(_load_dense_file(f, image_key, state_key, action_key))

    return episodes


def get_train_roots(
    data_root: Path,
    train_dir: str,
    left_dirname: str,
    right_dirname: str,
    variant: str,
) -> list[Path]:
    base = data_root / train_dir
    if variant == "right_only":
        return [base / right_dirname]
    if variant == "both":
        return [base / left_dirname, base / right_dirname]
    raise ValueError(f"Unknown dataset.variant: {variant}")


def get_test_roots(data_root: Path, test_dir: str) -> list[Path]:
    return [data_root / test_dir]


def compute_normalizer(episodes: list[Episode]) -> tuple[Normalizer, Normalizer]:
    states = np.concatenate([ep.state for ep in episodes], axis=0)
    actions = np.concatenate([ep.action for ep in episodes], axis=0)
    return (
        Normalizer(states.min(axis=0), states.max(axis=0)),
        Normalizer(actions.min(axis=0), actions.max(axis=0)),
    )


def summarize_episode_data(episodes: list[Episode]) -> dict:
    if len(episodes) == 0:
        return {"num_episodes": 0}
    image_shapes = np.asarray([ep.image.shape for ep in episodes], dtype=np.int64)
    state_shapes = np.asarray([ep.state.shape for ep in episodes], dtype=np.int64)
    action_shapes = np.asarray([ep.action.shape for ep in episodes], dtype=np.int64)
    return {
        "num_episodes": int(len(episodes)),
        "image_shape_example": tuple(episodes[0].image.shape),
        "state_shape_example": tuple(episodes[0].state.shape),
        "action_shape_example": tuple(episodes[0].action.shape),
        "seq_len_min": int(state_shapes[:, 0].min()),
        "seq_len_max": int(state_shapes[:, 0].max()),
        "state_dim": int(state_shapes[0, 1]),
        "action_dim": int(action_shapes[0, 1]),
        "channels": int(image_shapes[0, 1]),
        "height": int(image_shapes[0, 2]),
        "width": int(image_shapes[0, 3]),
    }


class WindowDataset(Dataset):
    def __init__(
        self,
        episodes: list[Episode],
        seq_len: int,
        state_norm: Normalizer,
        action_norm: Normalizer,
    ):
        self.seq_len = seq_len
        self.state_norm = state_norm
        self.action_norm = action_norm

        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for ep in episodes:
            if len(ep.action) < seq_len:
                continue
            n = len(ep.action) - seq_len + 1
            for i in range(n):
                img = ep.image[i : i + seq_len].astype(np.float32)
                st = self.state_norm.normalize(ep.state[i : i + seq_len])
                ac = self.action_norm.normalize(ep.action[i : i + seq_len])
                if img.max() > 1.0:
                    img = img / 255.0
                self.samples.append((img, st.astype(np.float32), ac.astype(np.float32)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, state, action = self.samples[idx]
        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(state).float(),
            torch.from_numpy(action).float(),
        )
