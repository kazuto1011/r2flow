<h1>How to reproduce R2Flow</h1>

This page provides instructions for training and evaluation procedures of R2Flow.

- [Training](#training)
  - [1. Initial training (`1-RF`)](#1-initial-training-1-rf)
  - [2. Straightening (`2-RF`)](#2-straightening-2-rf)
    - [2.1. Generate 1M samples from the `1-RF` model](#21-generate-1m-samples-from-the-1-rf-model)
    - [2.2 Finetune the `1-RF` model](#22-finetune-the-1-rf-model)
  - [3. $k$-timestep distillation (+ `k-TD`)](#3-k-timestep-distillation--k-td)
    - [3.1. Generate 1M samples from the `2-RF` model](#31-generate-1m-samples-from-the-2-rf-model)
    - [3.2 Finetune the `2-RF` model](#32-finetune-the-2-rf-model)
- [Evaluation](#evaluation)
- [Other settings](#other-settings)
  - [Model architecture](#model-architecture)

> [!NOTE]
> - This instruction uses `train.py`, `sample.py`, and `evaluate.py`.
> - Training and sampling are performed on distributed processes with available GPUs by default.
> - We tested this code on Ubuntu 22.04 + CUDA 12.1 + 4x NVIDIA RTX 6000 Ada GPUs.

## Training

R2Flow employs a [Rectified Flow](https://arxiv.org/abs/2209.03003)-based three-stage training as instructed below.

### 1. Initial training (`1-RF`)

To train an initial flow, run the following command:

```bash
accelerate launch --multi_gpu train.py \
    --batch_size 8 --loss_fn l2 --timestep_distribution uniform \
    --output_dir logs/r2flow-1rf
```

For usage of `accelerate launch`, please refer to [this page](https://huggingface.co/docs/accelerate/v1.8.1/en/basic_tutorials/launch#using-accelerate-launch).

The above command produces a result directory at `./logs/r2flow-1rf/kitti_360/spherical-1024/YYYYMMDDTHHMMSS` where `YYYYMMDDTHHMMSS` is the timestamp of the execution.

```sh
logs/
└── r2flow-1rf/
    └── kitti_360/                                 # Dataset
        └── spherical-1024/                        # Range image representation
            └── YYYYMMDDTHHMMSS/                   # Timestamp
                ├── models/                        # Saved checkpoints
                │   ├── checkpoint_0000080000.pth
                │   ├── ...
                │   └── checkpoint_0002560000.pth
                ├── events.out.tfevents.*          # TensorBoard events
                └── training_config.json           # Training configuration
```

To monitor the training progress, run the following command.
The `logs` directory can also be accessed from outside the Docker container.

```bash
tensorboard --logdir logs
```

### 2. Straightening (`2-RF`)

Assume that we have completed `1-RF` training under `./logs/r2flow-1rf/kitti_360/spherical-1024/YYYYMMDDTHHMMSS`.

#### 2.1. Generate 1M samples from the `1-RF` model

We use an adaptive ODE solver (`dopri5`) for preparing the training data of `2-RF`.
Note that the `dopri5` results may vary a bit depending on the batch size.

```bash
RESULT_DIR=./logs/r2flow-1rf/kitti_360/spherical-1024/YYYYMMDDTHHMMSS
CHECKPOINT_PATH=${RESULT_DIR}/models/checkpoint_0002560000.pth
SAMPLE_DIR=${RESULT_DIR}/samples_dopri5

accelerate launch --multi_gpu sample.py \
    --num_samples 1_000_000 --batch_size 64 \
    --solver dopri5 --tol 1e-5 \
    --ckpt ${CHECKPOINT_PATH} --output_dir ${SAMPLE_DIR}
```

The saved file manages a coupling `(x1, x0)` with the content and the filename.
For example, a file named with `sample_0000000001.pth` contains a data (`x1`) generated from a Gaussian noise (`x0`) with the random seed `int(0000000001)=1`.

#### 2.2 Finetune the `1-RF` model

```bash
accelerate launch --multi_gpu train.py \
    --init_ckpt ${CHECKPOINT_PATH} --sample_dir ${SAMPLE_DIR} \
    --batch_size 8 --loss_fn pseudo_huber --timestep_distribution u_shaped \
    --output_dir logs/r2flow-2rf
```

The resultant directories are like:

```sh
logs/
├── r2flow-1rf/
│   └── kitti_360/
│       └── spherical-1024/
│           └── YYYYMMDDTHHMMSS/
│               ├── models/
│               │   ├── ...
│               │   └── checkpoint_0002560000.pth  # Initial 1-RF weights
│               ├── ...
│               └── samples_dopri5/                # Generated 1M samples
│                   ├── ...
│                   └── sample_0000999999.pth
└── r2flow-2rf/
    └── kitti_360/
        └── spherical-1024/
            └── YYYYMMDDTHHMMSS/
                ├── models/                        # Saved checkpoints
                │   ├── ...
                │   └── checkpoint_0002560000.pth  # New 2-RF weights
                ├── events.out.tfevents.*
                └── training_config.json
```

### 3. $k$-timestep distillation (+ `k-TD`)

Assume that we have completed `2-RF` training under `./logs/r2flow-2rf/kitti_360/spherical-1024/YYYYMMDDTHHMMSS`.

#### 3.1. Generate 1M samples from the `2-RF` model

```bash
RESULT_DIR=./logs/r2flow-2rf/kitti_360/spherical-1024/YYYYMMDDTHHMMSS
CHECKPOINT_PATH=${RESULT_DIR}/models/checkpoint_0002560000.pth
SAMPLE_DIR=${RESULT_DIR}/samples_dopri5

accelerate launch --multi_gpu sample.py \
    --num_samples 1_000_000 --batch_size 64 \
    --solver dopri5 --tol 1e-5 \
    --ckpt ${CHECKPOINT_PATH} --output_dir ${SAMPLE_DIR}
```

#### 3.2 Finetune the `2-RF` model

```bash
for K_DISTIL in {1 2 4}; do
    accelerate launch --multi_gpu train.py \
        --ckpt ${CHECKPOINT_PATH} --sample_dir ${SAMPLE_DIR} \
        --batch_size 8 --loss_fn pseudo_huber --timestep_distribution uniform --k_distil ${K_DISTIL} \
        --output_dir logs/r2flow-2rf-${K_DISTIL}td
done
```

The resultant directories are like:

```sh
logs/
├── r2flow-1rf/
├── r2flow-2rf/
│   └── kitti_360/
│       └── spherical-1024/
│           └── YYYYMMDDTHHMMSS/
│               ├── models/
│               │   ├── ...
│               │   └── checkpoint_0002560000.pth  # Initial 2-RF weights
│               ├── ...
│               └── samples_dopri5/                # Generated 1M samples
├── r2flow-2rf-1td/  # 1-step distillation
├── r2flow-2rf-2td/  # 2-step distillation
└── r2flow-2rf-4td/  # 4-step distillation
    └── kitti_360/
        └── spherical-1024/
            └── YYYYMMDDTHHMMSS/
                ├── models/                        # Saved checkpoints
                │   ├── ...
                │   └── checkpoint_0002560000.pth  # New distilled weights
                ├── events.out.tfevents.*
                └── training_config.json
```

## Evaluation

We can evaluate the generated samples at any checkpoints. For example, if you want to evaluate the `1-RF` model with 32 timesteps, run the following commands:

```bash
NUM_STEPS=32
RESULT_DIR="logs/r2flow-1rf/kitti_360/spherical-1024/YYYYMMDDTHHMMSS"
CHECKPOINT_PATH=${RESULT_DIR}/models/checkpoint_0002560000.pth
SAMPLE_DIR=${RESULT_DIR}/samples_euler-${NUM_STEPS}

accelerate launch --multi_gpu sample.py \
    --num_samples 10_000 --batch_size 64 \
    --solver euler --num_steps ${NUM_STEPS} \
    --ckpt ${CHECKPOINT_PATH} \
    --output_dir ${SAMPLE_DIR}

python evaluate.py --sample_dir ${SAMPLE_DIR} \
    --export_name ${RESULT_DIR}/R2Flow-1RF-${NUM_STEPS}steps
```

The evaluation scores are saved as a json file:

```sh
logs/
└── r2flow-1rf/
    └── kitti_360/
        └── spherical-1024/
            └── YYYYMMDDTHHMMSS/
                ├── ...
                ├── models/
                │   └── checkpoint_0002560000.pth  # Target checkpoint
                ├── samples_euler-32/              # Generated 10k samples
                └── R2Flow-1RF-32steps.json        # Evaluation scores
```

The above commands report the following metrics.

|Metrics|Representation|Reflectance image|Range image|Point cloud|Voxel|Bird's eye view (BEV)|
|:-|:-|:-:|:-:|:-:|:-:|:-:|
|**FRD** (Fréchet range distance)<br>[Zyrianov et al. ECCV'22]|`RangeNet-53` feature|✓|✓||||
|**FRID** (Fréchet range image distance<br>[Ran et al. CVPR'24]|`RangeNet-21` feature||✓||||
|**FPD** (Fréchet point cloud distance)<br>[Shu et al. ICCV'19]|`PointNet` feature| | |✓| | |
|**FPVD** (Fréchet point volume distance)<br>[Ran et al. CVPR'24]|`SPVCNN` feature| | |✓|✓| |
|**FSVD** (Fréchet surface volume distance)<br>[Ran et al. CVPR'24]|`MinkowskiNet` feature| | ||✓| |
|**JSD** (Jensen–Shannon divergence)<br>[Zyrianov et al. ECCV'22]|Histogram| | | | |✓|
|**MMD** (maximum mean discrepancy)<br>[Zyrianov et al. ECCV'22]|Histogram| | | | |✓|

## Other settings

### Model architecture

To specify the model architecture used in our paper, add the following options.
All architectures below are modified to process the LiDAR imagery.

|Architecture|#Parameters|CLI options for `train.py`|
|:-|:-:|:-|
|**HDiT (default)**<br>[Crowson et al. ICML'24]|80.9M|`--architecture nat_hdit --base_channels 128 --temb_channels 256 --num_residual_blocks 3 3 3 3 --attn_num_head 2 4 8 16 --coords_encoding learnable_embedding`|
|**Efficient U-Net**<br>[Saharia et al. NeurIPS'22]|31.1M<br>(base)|`--architecture efficient_unet --base_channels 64 --channel_multiplier 1 2 4 8 --num_residual_blocks 3 3 3 3 --attn_num_head 8 --coords_encoding fourier_features`|
||284.6M<br>(large)|`--architecture efficient_unet --base_channels 128 --channel_multiplier 1 2 4 8 --num_residual_blocks 2 4 8 8 --attn_num_head 8 --coords_encoding fourier_features`|
|**ADM U-Net**<br>[Dhariwal et al. NeurIPS'21]|87.4M<br>(base)|`--architecture adm_unet --base_channels 64 --channel_multiplier 1 2 4 8 --num_residual_blocks 3 3 3 3 --attn_num_head 8 --coords_encoding fourier_features`|
||125.5M<br>(large)|`--architecture adm_unet --base_channels 128 --channel_multiplier 1 2 3 4 --num_residual_blocks 3 3 3 3 --attn_num_head 8 --coords_encoding fourier_features`|