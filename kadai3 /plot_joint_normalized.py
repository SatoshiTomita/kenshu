from __future__ import annotations

from pathlib import Path
import sys

import blosc2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from src.dataloader.episode_dataset import Normalizer  # type: ignore


def _load_blosc2_tensor(path: Path) -> np.ndarray:
    schunk = blosc2.open(str(path))
    _, shape, dtype = schunk.vlmeta[b"__pack_tensor__"]
    return np.frombuffer(schunk[:], dtype=np.dtype(dtype)).reshape(shape)


def _split_concat(arr: np.ndarray) -> list[np.ndarray]:
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [np.asarray(x) for x in arr.tolist()]
    if arr.ndim >= 3:
        return [np.asarray(x) for x in arr]
    return [np.asarray(arr)]


def _resolve_data_path(filename: str) -> Path:
    candidates = [
        ROOT / "data" / "right" / filename,
        ROOT / "src" / "data" / "right" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"{filename} not found in data/right or src/data/right")


def _plot_normalized(episodes: list[np.ndarray], title: str, out_name: str, ylabel_prefix: str) -> None:
    if not episodes:
        raise RuntimeError(f"No episodes found for {title}.")
    all_concat = np.concatenate(episodes, axis=0).astype(np.float32)
    norm = Normalizer(min=all_concat.min(axis=0), max=all_concat.max(axis=0))
    episodes_norm = [norm.normalize(ep.astype(np.float32)) for ep in episodes]

    n_dims = episodes_norm[0].shape[1]
    fig, axes = plt.subplots(nrows=n_dims, ncols=1, figsize=(12, 2 * n_dims), sharex=False)
    if n_dims == 1:
        axes = [axes]
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
    for d, ax in enumerate(axes):
        for i, ep in enumerate(episodes_norm):
            color = palette[(i // 15) % len(palette)]
            ax.plot(ep[:, d], linewidth=0.8, alpha=0.4, color=color)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(f"{ylabel_prefix} {d}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t (per episode)")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    out_path = ROOT / out_name
    fig.savefig(out_path, dpi=150)
    print({"saved": str(out_path), "episodes": len(episodes_norm)})


def main() -> None:
    joint_path = _resolve_data_path("joint_states.blosc2")
    action_path = _resolve_data_path("action_states.blosc2")

    joint_concat = _load_blosc2_tensor(joint_path)
    action_concat = _load_blosc2_tensor(action_path)

    joint_eps = _split_concat(joint_concat)
    action_eps = _split_concat(action_concat)

    _plot_normalized(
        episodes=joint_eps,
        title="Normalized Follower Joint States (right)",
        out_name="joint_states_normalized.png",
        ylabel_prefix="joint",
    )
    _plot_normalized(
        episodes=action_eps,
        title="Normalized Leader Action States (right)",
        out_name="action_states_normalized.png",
        ylabel_prefix="action",
    )


if __name__ == "__main__":
    main()
