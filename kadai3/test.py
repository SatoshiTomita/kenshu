from __future__ import annotations

from pathlib import Path
import sys
import time
import numpy as np
import hydra
import torch

_ROOT = Path(__file__).resolve().parent
sys.path.append(str(_ROOT / "src"))

from config_schema import MainConfig
from src.utils.train_utils import load_episodes, resolve_device, set_seed


def _build_replay_kwargs(cfg: MainConfig, Replay) -> dict:
    # Replay が受け取れる引数だけを渡す（バージョン差異に対応）
    import inspect

    params = set(inspect.signature(Replay).parameters.keys())
    # カメラ設定には触れない方針: 必須の解像度のみ渡す
    kwargs = {
        "height": cfg.replay.height,
        "width": cfg.replay.width,
        "camera_id": tuple(cfg.replay.camera_id),
        "is_higher_port": cfg.replay.is_higher_port,
        "leader_port": cfg.replay.leader_port,
        "follower_port": cfg.replay.follower_port,
        "calibration_name": cfg.replay.calibration_name,
    }
    return {k: v for k, v in kwargs.items() if k in params}


def run_dataset_replay(
    episodes: list[dict[str, np.ndarray]],
    send_action: bool,
    replay,
    fps: int,
    split: int,
    ema: float,
) -> dict:
    steps = 0
    sends = 0
    send_every = 1
    for ep_idx, ep in enumerate(episodes):
        action_seq = ep["action"]
        for t in range(len(action_seq)):
            action_t = torch.tensor(action_seq[t], dtype=torch.float32)

            if send_action and (steps % send_every == 0):
                replay.send(action=action_t, fps=float(fps), split=split, ema=ema)
                sends += 1
                time.sleep(1.0 / float(fps))
            steps += 1

    return {"episodes": len(episodes), "steps": int(steps), "sends": int(sends)}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: MainConfig):
    set_seed(int(cfg.seed))
    _ = resolve_device(cfg.device)

    train_episodes, test_episodes = load_episodes(cfg)
    episodes = test_episodes if len(test_episodes) > 0 else train_episodes

    if cfg.replay.send_action:
        from lerobot_utils import Replay

        replay = Replay(**_build_replay_kwargs(cfg, Replay))
    else:
        replay = None

    result = run_dataset_replay(
        episodes=episodes,
        send_action=bool(cfg.replay.send_action),
        replay=replay,
        fps=int(cfg.replay.fps),
        split=int(cfg.replay.split),
        ema=float(cfg.replay.ema),
    )
    print(result)


if __name__ == "__main__":
    main()
