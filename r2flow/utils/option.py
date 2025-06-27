from typing import List, Literal, Tuple

from pydantic.dataclasses import dataclass
from simple_parsing.helpers import list_field


@dataclass
class ModelConfig:
    architecture: str = "nat_hdit"
    base_channels: int = 128
    temb_channels: int | None = 256
    channel_multiplier: List[int] = list_field(1, 2, 4, 8)
    num_residual_blocks: List[int] = list_field(3, 3, 3, 3)
    gn_num_groups: int = 32 // 4
    gn_eps: float = 1e-6
    attn_num_heads: List[int] | int = list_field(2, 4, 8, 16)
    coords_encoding: Literal[
        "spherical_harmonics",
        "polar_coordinates",
        "fourier_features",
        "learnable_embedding",
        None,
    ] = "learnable_embedding"
    dropout: float = 0.0


@dataclass
class FlowConfig:
    num_sampling_steps: int = 32
    formulation: Literal["otcfm", "icfm"] = "icfm"
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
    mixed_precision: str = "no"
    dynamo_backend: str = "inductor"
    output_dir: str = "logs/r2flow-1rf"
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
class FinetuningConfig:
    init_ckpt: str | None = None
    sample_dir: str | None = None
    k_distil: int | None = None


@dataclass
class DefaultConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    flow: FlowConfig = FlowConfig()
    training: TrainingConfig = TrainingConfig()
    finetuning: FinetuningConfig = FinetuningConfig()
