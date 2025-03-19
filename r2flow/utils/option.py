from typing import List, Literal, Tuple

from pydantic.dataclasses import dataclass
from simple_parsing.helpers import list_field


@dataclass
class ModelConfig:
    architecture: str = "efficient_unet"
    base_channels: int = 64
    temb_channels: int | None = None
    channel_multiplier: List[int] = list_field(1, 2, 4, 8)
    num_residual_blocks: List[int] = list_field(3, 3, 3, 3)
    gn_num_groups: int = 32 // 4
    gn_eps: float = 1e-6
    attn_num_heads: int = 8
    coords_encoding: Literal[
        "spherical_harmonics",
        "polar_coordinates",
        "fourier_features",
        "learnable_embedding",
        None,
    ] = "fourier_features"
    dropout: float = 0.0


@dataclass
class FlowConfig:
    num_sampling_steps: int = 32
    formulation: Literal["otcfm", "icfm", "sbcfm", "1rf", "fm"] = "otcfm"
    sigma: float = 0.0
    solver: str = "dopri5"


@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_workers: int = 4
    num_images_training: int = 2_560_000  # ~= 300_000 iters * 8
    num_images_lr_warmup: int = 80_000  # ~= 10_000 iters * 8
    num_steps_logging: int = 1_000
    num_steps_checkpoint: int = 10_000  # 1/32
    gradient_accumulation_steps: int = 1
    lr: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    ema_decay: float = 0.995
    ema_update_every: int = 10
    mixed_precision: str = "fp16"
    dynamo_backend: str = "inductor"
    output_dir: str = "logs/flow"
    seed: int = 0
    timestep_distribution: Literal["uniform", "u_shaped", "logit_normal"] = "uniform"
    loss_fn: Literal["l1", "l2", "smooth_l1", "pseudo_huber"] = "l2"


@dataclass
class DataConfig:
    dataset: Literal["kitti_raw", "kitti_360"] = "kitti_360"
    data_format: Literal["logscale", "inverse", "metric", "cartesian"] = "logscale"
    train_reflectance: bool = True
    projection: Literal[
        "unfolding-2048",
        "spherical-2048",
        "unfolding-1024",
        "spherical-1024",
    ] = "spherical-1024"
    resolution: Tuple[int, int] = (64, 1024)
    min_depth: float = 1.45
    max_depth: float = 80.0


@dataclass
class DefaultConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    flow: FlowConfig = FlowConfig()
    training: TrainingConfig = TrainingConfig()


@dataclass
class ReFlowConfig:
    training: TrainingConfig = TrainingConfig(
        batch_size=8,
        num_workers=4,
        num_images_training=2_560_000,
        num_images_lr_warmup=80_000,
        num_steps_logging=1_000,
        num_steps_checkpoint=10_000,
        gradient_accumulation_steps=1,
        lr=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_weight_decay=0.0,
        adam_epsilon=1e-8,
        ema_decay=0.995,
        ema_update_every=10,
        mixed_precision="fp16",
        dynamo_backend="inductor",
        output_dir="logs/reflow",
        seed=0,
        timestep_distribution="u_shaped",
        loss_fn="pseudo_huber",
    )
