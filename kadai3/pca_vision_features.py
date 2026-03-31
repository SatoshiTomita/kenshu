"""
学習済み VisionPolicyModel の vision 特徴量に対して PCA を行う。

入力画像は image_states.blosc2 のみを用いる（学習時の action/joint は不要）。

使用例（kadai3 ディレクトリから）::

    uv run python pca_vision_features.py --model-dir result/my_run

    # ../result/run と指定した場合、kadai3/result/run にフォールバックする
    #
    # t=0 のみ（エピソード数=点数）:
    #   uv run python pca_vision_features.py --model-dir result/my_run --time-index 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

_KADAI3 = Path(__file__).resolve().parent
# import src.* 用と config_schema 直 import 用（train_utils）
sys.path.insert(0, str(_KADAI3))
sys.path.insert(0, str(_KADAI3 / "src"))

from src.dataloader.episode_dataset import (  # noqa: E402
    _ensure_image_seq,
    _load_blosc2_tensor,
    _split_concat,
)
from src.utils.app_helpers import build_model  # noqa: E402
from src.utils.train_utils import load_cfg, load_normalizers, resolve_device, set_seed  # noqa: E402


def _match_in_channels(image: np.ndarray, in_channels: int) -> np.ndarray:
    """[T,C,H,W] をモデルの in_channels に合わせる。"""
    c = image.shape[1]
    if c == in_channels:
        return image
    if c == 1 and in_channels == 3:
        return np.repeat(image, 3, axis=1)
    if c == 3 and in_channels == 1:
        return image[:, :1, :, :]
    raise ValueError(f"Cannot map image channels {c} to vision.in_channels={in_channels}")


def _grouped_episode_boundaries(n_ep: int, step: int = 15) -> np.ndarray:
    """エピソード番号 0..n_ep-1 を step 個ごとに区切る BoundaryNorm 用境界（端数も1区間）。"""
    n_ep = max(int(n_ep), 1)
    edges = np.arange(-0.5, n_ep + 0.5, float(step))
    if edges[-1] < n_ep - 0.5:
        edges = np.append(edges, n_ep - 0.5)
    return edges


def _discrete_cmap_n_colors(n: int) -> ListedColormap:
    """n 色の離散カラーマップ（tab20 を繰り返し、グラデーションなし）。"""
    n = max(int(n), 1)
    base = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = np.vstack([base] * ((n + 19) // 20))[:n]
    return ListedColormap(colors)


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


@torch.no_grad()
def collect_vision_features(
    model: torch.nn.Module,
    episodes: list[np.ndarray],
    device: torch.device,
    in_channels: int,
    batch_frames: int,
    max_frames: int,
    time_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    各フレームの vision 特徴を [N, F] で返す。labels はエピソードインデックス [N]。

    time_index が None のときは全フレーム。整数のときは各エピソードでその時刻の1枚だけ
    （例: 0 なら t=0 のみ）→ N はエピソード数以下になる。
    """
    model.eval()
    feats_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    def flush(buf_img: list[np.ndarray], buf_lbl: list[np.ndarray]) -> None:
        if not buf_img:
            return
        x = np.stack(buf_img, axis=0)
        lbl = np.concatenate(buf_lbl, axis=0)
        t = torch.from_numpy(x).unsqueeze(0).to(device)
        feat = model.vision(t)
        feat = feat.squeeze(0).cpu().numpy()
        feats_list.append(feat)
        labels_list.append(lbl)

    buf_img: list[np.ndarray] = []
    buf_lbl: list[np.ndarray] = []
    total = 0

    for ep_idx, image in enumerate(tqdm(episodes, desc="episodes")):
        image = _match_in_channels(image, in_channels)
        tlen = image.shape[0]
        if time_index is not None:
            if tlen <= int(time_index):
                continue
            frame_indices = [int(time_index)]
        else:
            frame_indices = range(tlen)
        for i in frame_indices:
            if max_frames > 0 and total >= max_frames:
                flush(buf_img, buf_lbl)
                return np.concatenate(feats_list, axis=0), np.concatenate(labels_list, axis=0)
            buf_img.append(image[i])
            buf_lbl.append(np.array([ep_idx], dtype=np.int64))
            total += 1
            if len(buf_img) >= batch_frames:
                flush(buf_img, buf_lbl)
                buf_img = []
                buf_lbl = []

    flush(buf_img, buf_lbl)
    if not feats_list:
        raise RuntimeError("No frames collected; check image_states.blosc2 and max_frames.")
    return np.concatenate(feats_list, axis=0), np.concatenate(labels_list, axis=0)


def resolve_model_dir(arg: Path, kadai3: Path) -> Path:
    """
    学習結果ディレクトリを決定する。
    kadai3 から ../result/run と指定するとリポジトリ直下の result を指しがちなので、
    kadai3/result/run があればそちらを優先する。
    """
    seen: set[str] = set()
    candidates: list[Path] = []

    def add(p: Path) -> None:
        p = p.resolve()
        s = str(p)
        if s not in seen:
            seen.add(s)
            candidates.append(p)

    add(arg)
    if not arg.is_absolute():
        add(kadai3 / arg)
    if arg.name:
        add(kadai3 / "result" / arg.name)
    resolved = arg.resolve()
    if resolved.parent.name == "result" and resolved.name:
        add(kadai3 / "result" / resolved.name)

    for c in candidates:
        if (c / "config.yaml").is_file():
            return c
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"学習結果に config.yaml が見つかりません。試したパス: {tried}"
    )


def main() -> None:
    default_image = _KADAI3 / "src/data/right_tmp/image_states.blosc2"
    parser = argparse.ArgumentParser(description="PCA on vision features from a trained model.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="学習結果ディレクトリ（config.yaml, normalizer.yaml, best_model.pt を含む）",
    )
    parser.add_argument(
        "--image-blosc2",
        type=Path,
        default=default_image,
        help="読み込む image_states.blosc2 のパス（既定: kadai3/src/data/right/image_states.blosc2）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-components", type=int, default=2, help="PCA の主成分数（散布図は先頭2成分）")
    parser.add_argument("--max-frames", type=int, default=0, help="0 なら全フレーム")
    parser.add_argument(
        "--time-index",
        type=int,
        default=None,
        help="各エピソードでこの時刻の1枚だけ使う（例: 0 で t=0 のみ → 点数はエピソード数）。未指定なら全時刻",
    )
    parser.add_argument("--batch-frames", type=int, default=256, help="GPU に載せるフレーム数（vision は B=1, S=batch）")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="出力先（既定: <model-dir>/pca_vision）",
    )
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir, _KADAI3)
    cfg_path = model_dir / "config.yaml"
    norm_path = model_dir / "normalizer.yaml"
    model_path = model_dir / "best_model.pt"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not norm_path.is_file():
        raise FileNotFoundError(norm_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    set_seed(int(args.seed))
    device = resolve_device("auto")
    cfg = load_cfg(cfg_path)
    state_norm, action_norm = load_normalizers(norm_path)
    state_dim = len(state_norm.min)
    action_dim = len(action_norm.min)

    model = build_model(cfg, state_dim=state_dim, action_dim=action_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    image_path = (
        args.image_blosc2.resolve()
        if args.image_blosc2.is_absolute()
        else (_KADAI3 / args.image_blosc2).resolve()
    )
    episodes = load_image_episodes_from_blosc2(image_path)
    in_ch = int(cfg.vision.in_channels)

    feats, ep_labels = collect_vision_features(
        model=model,
        episodes=episodes,
        device=device,
        in_channels=in_ch,
        batch_frames=int(args.batch_frames),
        max_frames=int(args.max_frames),
        time_index=args.time_index,
    )

    n_comp = min(int(args.n_components), feats.shape[1], feats.shape[0])
    pca = PCA(n_components=n_comp, random_state=int(args.seed))
    z = pca.fit_transform(feats)

    out_dir = args.out_dir if args.out_dir is not None else (model_dir / "pca_vision")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_kw: dict = {
        "features": feats.astype(np.float32),
        "embedding": z.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "episode_index": ep_labels,
    }
    if args.time_index is not None:
        save_kw["time_index"] = np.int64(args.time_index)
    np.savez_compressed(out_dir / "pca_result.npz", **save_kw)

    ev_path = out_dir / "explained_variance_ratio.txt"
    with ev_path.open("w") as f:
        for i, r in enumerate(pca.explained_variance_ratio_):
            f.write(f"PC{i+1}: {r:.6f}\n")

    if z.shape[1] >= 2:
        plt.figure(figsize=(7, 6))
        if args.time_index is not None:
            # 15 エピソードごとに同色。カラーバーは 0〜n_ep-1 の目盛り（離散、グラデーションなし）
            ep_int = ep_labels.astype(np.float64)
            n_ep = int(ep_labels.max()) + 1
            boundaries = _grouped_episode_boundaries(n_ep, step=15)
            n_grp = len(boundaries) - 1
            cmap_lc = _discrete_cmap_n_colors(n_grp)
            norm = BoundaryNorm(boundaries, cmap_lc.N)
            sc = plt.scatter(
                z[:, 0],
                z[:, 1],
                c=ep_int,
                s=24,
                alpha=0.85,
                cmap=cmap_lc,
                norm=norm,
            )
            tick_vals = np.arange(0, n_ep + 1, 15)
            cbar = plt.colorbar(sc, label="episode index", ticks=tick_vals)
            title_suffix = f" (t={args.time_index} only)"
        else:
            color_labels = np.arange(z.shape[0], dtype=np.int64) // 15
            sc = plt.scatter(
                z[:, 0],
                z[:, 1],
                c=color_labels,
                s=4,
                alpha=0.7,
                cmap="tab20",
            )
            plt.colorbar(sc, label="color group (every 15 frames)")
            title_suffix = ""
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"Vision features PCA (image_states.blosc2){title_suffix}")
        plt.tight_layout()
        plt.savefig(out_dir / "pca_scatter_pc1_pc2.png", dpi=150)
        plt.close()

    print(
        {
            "model_dir": str(model_dir),
            "image_blosc2": str(image_path),
            "num_episodes": len(episodes),
            "num_frames": int(feats.shape[0]),
            "time_index": args.time_index,
            "feature_dim": int(feats.shape[1]),
            "out_dir": str(out_dir),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }
    )


if __name__ == "__main__":
    main()
