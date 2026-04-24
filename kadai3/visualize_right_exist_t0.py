"""
right_exist の t=0 画像を 30 枚ずつ可視化して保存する。

使用例（kadai3 ディレクトリから）::

    uv run python visualize_right_exist_t0.py
    uv run python visualize_right_exist_t0.py --per-page 30 --max-episodes 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_KADAI3 = Path(__file__).resolve().parent
sys.path.insert(0, str(_KADAI3))
sys.path.insert(0, str(_KADAI3 / "src"))

from src.dataloader.episode_dataset import _ensure_image_seq, _load_blosc2_tensor, _split_concat


def load_image_episodes_from_blosc2(path: Path) -> list[np.ndarray]:
    """image_states.blosc2 からエピソードごとの [T,C,H,W] を返す。"""
    if not path.is_file():
        raise FileNotFoundError(f"image_states not found: {path}")
    raw = _load_blosc2_tensor(path)
    chunks = _split_concat(raw)
    out: list[np.ndarray] = []
    for chunk in chunks:
        img = _ensure_image_seq(np.asarray(chunk))
        out.append(img.astype(np.float32))
    return out


def _to_imshow(image_chw: np.ndarray) -> np.ndarray:
    """[C,H,W] -> imshow 用 [H,W] or [H,W,3] に変換。"""
    if image_chw.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {image_chw.shape}")
    c, h, w = image_chw.shape
    if c == 1:
        img = image_chw[0]
    else:
        img = image_chw[:3].transpose(1, 2, 0)
    # 0-255 の可能性があるので軽く正規化
    mx = float(np.max(img))
    if mx > 1.0:
        img = img / 255.0
    return img


def main() -> None:
    default_image = _KADAI3 / "src/data/right_exist/image_states.blosc2"
    parser = argparse.ArgumentParser(description="Visualize right_exist t=0 images (30 per page).")
    parser.add_argument(
        "--image-blosc2",
        type=Path,
        default=default_image,
        help="読み込む image_states.blosc2 のパス（既定: kadai3/src/data/right_exist/image_states.blosc2）",
    )
    parser.add_argument("--time-index", type=int, default=0, help="可視化する時刻（既定: 0）")
    parser.add_argument("--per-page", type=int, default=30, help="1ページに並べる枚数（既定: 30）")
    parser.add_argument("--max-episodes", type=int, default=0, help="0 なら全エピソード")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_KADAI3 / "outputs" / "right_exist_t0",
        help="出力先ディレクトリ",
    )
    args = parser.parse_args()

    image_path = (
        args.image_blosc2.resolve()
        if args.image_blosc2.is_absolute()
        else (_KADAI3 / args.image_blosc2).resolve()
    )
    episodes = load_image_episodes_from_blosc2(image_path)
    if args.max_episodes > 0:
        episodes = episodes[: int(args.max_episodes)]

    t_idx = int(args.time_index)
    per_page = max(int(args.per_page), 1)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images: list[np.ndarray] = []
    for ep_idx, ep in enumerate(episodes):
        if ep.shape[0] <= t_idx:
            continue
        images.append(ep[t_idx])

    if not images:
        raise RuntimeError("No images collected. Check time-index and dataset.")

    n_pages = (len(images) + per_page - 1) // per_page
    ncols = 6
    nrows = int(np.ceil(per_page / ncols))

    for page_idx in range(n_pages):
        batch = images[page_idx * per_page : (page_idx + 1) * per_page]
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
        axes = np.array(axes).reshape(-1)

        for ax_i, ax in enumerate(axes):
            ax.axis("off")
            if ax_i >= len(batch):
                continue
            img = _to_imshow(batch[ax_i])
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(img, vmin=0.0, vmax=1.0)
            ax.set_title(f"{page_idx * per_page + ax_i}", fontsize=8)

        fig.suptitle(f"right_exist t={t_idx} (page {page_idx + 1}/{n_pages})", fontsize=12)
        fig.tight_layout()
        out_path = out_dir / f"right_exist_t{t_idx}_{page_idx:03d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print(
        {
            "image_blosc2": str(image_path),
            "num_episodes": len(episodes),
            "num_images": len(images),
            "time_index": t_idx,
            "per_page": per_page,
            "out_dir": str(out_dir),
            "pages": n_pages,
        }
    )


if __name__ == "__main__":
    main()
