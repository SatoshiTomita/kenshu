from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.dataloader.episode_dataset import (
    _ensure_image_seq,
    _load_blosc2_tensor,
    _split_concat,
)


def _resolve_image_blosc2(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        Path("src/data/right/image_states.blosc2"),
        Path("src/data/right_tmp/image_states.blosc2"),
        Path("src/data/right_exist/image_states.blosc2"),
        Path("data/right/image_states.blosc2"),
    ]
    tried = [path]
    for c in candidates:
        if c.is_file():
            return c
        tried.append(c)
    tried_msg = ", ".join(str(p) for p in tried)
    raise FileNotFoundError(f"image_states.blosc2 not found. Tried: {tried_msg}")


def _to_display_image(img_t: np.ndarray) -> np.ndarray:
    if img_t.ndim != 3:
        raise ValueError(f"Expected image [C,H,W], got {img_t.shape}")
    c, h, w = img_t.shape
    if c not in (1, 3):
        raise ValueError(f"Unsupported channels: {c}")
    img = np.transpose(img_t, (1, 2, 0))
    if img.max() <= 1.0:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    else:
        img = img.clip(0, 255).astype(np.uint8)
    if c == 1:
        return img[:, :, 0]
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a single t=0 image frame.")
    parser.add_argument(
        "--image-blosc2",
        type=Path,
        default=Path("image_states.blosc2"),
        help="image_states.blosc2 path (default: ./image_states.blosc2 or fallback candidates)",
    )
    parser.add_argument("--episode-index", type=int, default=0, help="episode index")
    parser.add_argument("--time-index", type=int, default=0, help="time index (t)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output image path (default: ./t0_image_ep{ep}_t{t}.png)",
    )
    parser.add_argument("--show", action="store_true", help="display window with matplotlib")
    args = parser.parse_args()

    image_path = _resolve_image_blosc2(args.image_blosc2)
    raw = _load_blosc2_tensor(image_path)
    episodes = _split_concat(raw)
    if not episodes:
        raise RuntimeError("No image episodes found in image_states.blosc2")
    ep_idx = int(args.episode_index)
    if ep_idx < 0 or ep_idx >= len(episodes):
        raise IndexError(f"episode index out of range: {ep_idx} (0..{len(episodes)-1})")
    image_seq = _ensure_image_seq(np.asarray(episodes[ep_idx]))

    t = int(args.time_index)
    if t < 0 or t >= image_seq.shape[0]:
        raise IndexError(f"time index out of range: {t} (0..{image_seq.shape[0]-1})")

    img_t = image_seq[t]
    disp = _to_display_image(img_t)

    out_path = args.out
    if out_path is None:
        out_path = Path.cwd() / f"t0_image_ep{ep_idx}_t{t}.png"

    plt.figure(figsize=(6, 4))
    if disp.ndim == 2:
        plt.imshow(disp, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(disp)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print({"saved": str(out_path), "image_blosc2": str(image_path), "episode": ep_idx, "t": t})

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
