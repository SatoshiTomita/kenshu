from __future__ import annotations

import argparse
from pathlib import Path

import blosc2
import numpy as np
import matplotlib.pyplot as plt


def _extract_sort_key(path: Path) -> tuple[int, str]:
    label = path.stem
    try:
        return int(label.split("_")[-1]), label
    except ValueError:
        return 10**9, label


def _load_blosc2_tensor(path: Path) -> np.ndarray:
    schunk = blosc2.open(str(path))
    _, shape, dtype = schunk.vlmeta[b"__pack_tensor__"]
    return np.frombuffer(schunk[:], dtype=np.dtype(dtype)).reshape(shape)


def _load_batch(paths: list[Path]) -> np.ndarray:
    series_list = [_load_blosc2_tensor(path) for path in paths]
    first_shape = series_list[0].shape
    for path, series in zip(paths, series_list):
        if series.shape != first_shape:
            raise ValueError(f"Shape mismatch in {path}: {series.shape} != {first_shape}")
    return np.stack(series_list, axis=0)  # [N, T, D]


def _normalize_batch_to_minus1_1(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = batch.min(axis=(0, 1), keepdims=True)  # [1,1,D]
    maxs = batch.max(axis=(0, 1), keepdims=True)  # [1,1,D]
    denom = np.where((maxs - mins) < 1e-8, 1.0, (maxs - mins))
    normalized = 2.0 * (batch - mins) / denom - 1.0
    return normalized, mins.squeeze(0).squeeze(0), maxs.squeeze(0).squeeze(0)


def create_joint_action_state_grid(
    raw_dir: Path,
    output_path: Path,
    max_items: int = 31,
    use: str = "action",
    dim: int = 0,
    all_dims: bool = False,
    normalize: bool = True,
):
    files = sorted(raw_dir.glob(f"{use}_*.blosc2"), key=_extract_sort_key)
    if not files:
        raise RuntimeError(f"No {use}_*.blosc2 found in: {raw_dir}")

    selected = files[:max_items]
    batch = _load_batch(selected)
    first = batch[0]
    if first.ndim != 2:
        raise ValueError(f"Expected 2D tensor in {selected[0]}, got shape={first.shape}")
    dims = first.shape[1]
    if normalize:
        batch, dim_mins, dim_maxs = _normalize_batch_to_minus1_1(batch)
    else:
        dim_mins = np.min(batch, axis=(0, 1))
        dim_maxs = np.max(batch, axis=(0, 1))

    if all_dims:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
        axes = np.asarray(axes).reshape(-1)
        for i, path in enumerate(selected):
            series = batch[i]
            label = path.stem.replace(f"{use}_", "")
            for d in range(min(dims, 6)):
                axes[d].plot(series[:, d], linewidth=1.0, alpha=0.75, label=label)

        for d in range(6):
            axes[d].set_title(f"{use} dim={d} min={dim_mins[d]:.3f} max={dim_maxs[d]:.3f}")
            axes[d].set_xlabel("time step")
            axes[d].set_ylabel("value")
            axes[d].grid(alpha=0.25)
            if normalize:
                axes[d].set_ylim(-1.05, 1.05)
        axes[0].legend(ncol=6, fontsize=6, frameon=False, loc="upper right")
        fig.suptitle(
            f"{use} dim0-5 overlay ({len(selected)} lines each)"
            + (" normalized to [-1,1]" if normalize else ""),
            fontsize=13,
        )
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        if not (0 <= dim < dims):
            raise ValueError(f"dim={dim} out of range. available: 0..{dims-1}")
        for i, path in enumerate(selected):
            series = batch[i]
            label = path.stem.replace(f"{use}_", "")
            ax.plot(series[:, dim], linewidth=1.0, alpha=0.8, label=label)

        ax.set_title(
            f"{use} dim={dim} overlay ({len(selected)} lines) "
            f"min={dim_mins[dim]:.3f} max={dim_maxs[dim]:.3f}"
        )
        ax.set_xlabel("time step")
        ax.set_ylabel("value")
        ax.grid(alpha=0.25)
        if normalize:
            ax.set_ylim(-1.05, 1.05)
        ax.legend(ncol=6, fontsize=7, frameon=False, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parse_args():
    parser = argparse.ArgumentParser(description="Create one line chart that overlays 30 sequences.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("src/data/right/raw"),
        help="Directory containing state_*.blosc2 and action_*.blosc2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("result/right_joint_action_state_lines_30.png"),
        help="Path to save overlay line chart.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="",
        help="If set, save to result/<output-name> instead of --output.",
    )
    parser.add_argument("--max-items", type=int, default=30, help="Number of files to include in the grid.")
    parser.add_argument("--use", choices=("state", "action"), default="state", help="Which sequence to plot.")
    parser.add_argument("--dim", type=int, default=0, help="Dimension index to plot as one line per sample.")
    parser.add_argument("--all-dims", action="store_true", help="Plot dim0-5 in one figure (2x3 subplots).")
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize each dimension to [-1,1] using min/max over selected samples.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_path = Path("result") / args.output_name if args.output_name else args.output
    create_joint_action_state_grid(
        raw_dir=args.raw_dir,
        output_path=output_path,
        max_items=args.max_items,
        use=args.use,
        dim=args.dim,
        all_dims=args.all_dims,
        normalize=args.normalize,
    )
    print(f"Saved: {output_path}")
