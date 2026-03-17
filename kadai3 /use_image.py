from __future__ import annotations

from pathlib import Path
import sys

import blosc2
import hydra
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
sys.path.append(str(_ROOT / "src"))

from src.utils.app_helpers import build_model
from src.utils.train_utils import load_normalizers, resolve_device, set_seed


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


def _resolve_data_root(data_root: Path) -> Path:
    if data_root.exists():
        return data_root
    alt = Path("src") / data_root
    if alt.exists():
        return alt
    return data_root


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)

    result_dir = Path(cfg.result_root) / cfg.train_name
    state_norm, action_norm = load_normalizers(result_dir / "normalizer.yaml")

    model = build_model(
        cfg,
        state_dim=len(state_norm.min),
        action_dim=len(action_norm.min),
    ).to(device)
    model_path = Path(cfg.replay.model_path) if cfg.replay.model_path else result_dir / "best_model.pt"
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    base = _resolve_data_root(Path(cfg.dataset.root))
    if cfg.dataset.variant == "left_only":
        base = base / cfg.dataset.left_dirname
    elif cfg.dataset.variant == "right_only":
        base = base / cfg.dataset.right_dirname
    else:
        base = base / cfg.dataset.right_dirname
    image_path = base / cfg.dataset.image_states_filename
    if not image_path.exists():
        raise RuntimeError(f"image_states not found: {image_path}")
    image_all = _load_blosc2_tensor(image_path)
    image_eps = _split_concat(image_all)
    if len(image_eps) == 0:
        raise RuntimeError("No image episodes found in image_states.blosc2")
    image_seq = _ensure_image_seq(image_eps[0])

    from lerobot_utils import Replay  # type: ignore

    replay = Replay(
        height=480,
        width=640,
        camera_id=(0,),
        is_higher_port=False,
        leader_port="/dev/tty.usbmodem58370530001",
        follower_port="/dev/tty.usbmodem58370529971",
        calibration_name="koch",
    )

    model.eval()
    image_q: list[torch.Tensor] = []
    state_q: list[np.ndarray] = []

    steps = int(cfg.replay.steps)
    for t in range(steps):
        idx = t % int(image_seq.shape[0])
        obs = replay.get_observations(max_depth=cfg.replay.max_depth)
        state = np.asarray(obs[cfg.replay.state_key], dtype=np.float32)

        image_t = torch.tensor(image_seq[idx], dtype=torch.float32)
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
            replay.send(action=action_t, fps=cfg.replay.fps)

    print({"mode": "use_image", "steps": int(steps), "send_action": bool(cfg.replay.send_action)})


if __name__ == "__main__":
    main()
