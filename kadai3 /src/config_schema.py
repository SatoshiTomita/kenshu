from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WandbConfig:
    enable: bool = True
    project: str = "kadai3"
    run_name: str = "${train_name}"
    config: dict = field(default_factory=lambda: {"note": "cnn+rnn baseline"})


@dataclass
class DatasetConfig:
    root: str = "data"
    train_dir: str = "train"
    test_dir: str = "test"
    left_dirname: str = "left"
    right_dirname: str = "right"
    variant: str = "both"
    action_states_filename: str = "action_states.blosc2"
    image_states_filename: str = "image_states.blosc2"
    joint_states_filename: str = "joint_states.blosc2"
    image_key: str = "image"
    state_key: str = "state"
    action_key: str = "action"
    seq_len: int = 8
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    # 画像データ拡張の設定
    augment: bool = False
    aug_brightness: float = 0.1
    aug_contrast: float = 0.1
    aug_shift: int = 4
    aug_crop: int = 0


@dataclass
class VisionConfig:
    in_channels: int = 3
    conv_channels: list[int] = field(default_factory=lambda: [16, 32, 64])
    kernels: list[int] = field(default_factory=lambda: [5, 3, 3])
    strides: list[int] = field(default_factory=lambda: [2, 2, 2])
    paddings: list[int] = field(default_factory=lambda: [2, 1, 1])
    feature_dim: int = 128


@dataclass
class PolicyConfig:
    rnn_type: str = "rnn"
    hidden_dim: int = 128
    num_layers: int = 1
    action_horizon: int = 1


@dataclass
class TrainerConfig:
    epochs: int = 40
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    grad_clip_norm: float = 1.0
    use_state: bool = True
    state_noise_std: float = 0.0


@dataclass
class ReplayConfig:
    enable: bool = True
    model_path: str = ""
    model_name: str = ""
    steps: int = 150
    fps: int = 30
    ema: float = 0.0
    split: int = 100
    send_action: bool = True
    image_key: str = "image.main"
    state_key: str = "observation.state"
    max_depth: float | None = None
    init_from_current_state: bool = True
    height: int = 480
    width: int = 640
    resize_height: int | None = None
    resize_width: int | None = None
    serial_numbers: list[str] = field(default_factory=list)
    camera_id: list[int] = field(default_factory=lambda: [0])
    scale: float = 1.0
    auto_exposure: bool = False
    auto_white_balance: bool = False
    exposure: float = 190.0
    white_balance: float = 3300.0
    min_depth: float = 0.0
    max_depth_realsense: float = 1500.0
    is_higher_port: bool = False
    leader_port: str = "/dev/tty.usbmodem58370530001"
    follower_port: str = "/dev/tty.usbmodem58370529971"
    calibration_name: str = "koch"


@dataclass
class HydraRunConfig:
    dir: str = "."


@dataclass
class HydraJobConfig:
    chdir: bool = False


@dataclass
class HydraConfig:
    run: HydraRunConfig = field(default_factory=HydraRunConfig)
    output_subdir: str | None = None
    job: HydraJobConfig = field(default_factory=HydraJobConfig)


@dataclass
class MainConfig:
    mode: str = "offline_test"
    train_name: str = "cnn_rnn_baseline"
    seed: int = 42
    result_root: str = "result"
    device: str = "auto"
    wandb: WandbConfig = field(default_factory=WandbConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    hydra: HydraConfig = field(default_factory=HydraConfig)
