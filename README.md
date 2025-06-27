# R2Flow

* This is the official implementation of our **ICRA 2025** paper, **"Fast LiDAR Data Generation with Rectified Flows"**.
* **R2Flow** is a rectified flow-based LiDAR generative model, which can generate LiDAR data in few steps.

https://github.com/user-attachments/assets/5fdb9469-799b-438f-8334-a16d5e8180f8

**Fast LiDAR Data Generation with Rectified Flows**<br>
[Kazuto Nakashima](https://kazuto1011.github.io), Xiaowen Liu, Tomoya Miyawaki, Yumi Iwashita, Ryo Kurazume<br>
ICRA 2025<br>
[Project page](https://kazuto1011.github.io/r2flow) | [arXiv](https://arxiv.org/abs/2412.02241) | [Demo](https://huggingface.co/spaces/kazuto1011/r2flow)

## Setup

### Dataset (optional)

For training and evaluation, download a [KITTI-360 dataset](http://www.cvlibs.net/datasets/kitti-360/) (163 GB) and set an environment variable `${KITTI360_ROOT}`:

```bash
export KITTI360_ROOT=<path to the KITTI-360 dataset>
```

```sh
KITTI360_ROOT
├── data_3d_raw/
│   ├── 2013_05_28_drive_0000_sync
│   └── ...
└── data_3d_semantics/
    ├── 2013_05_28_drive_0000_sync
    └── ...
```

### Dependencies

Clone this repository:

```bash
git clone https://github.com/kazuto1011/r2flow.git
cd r2flow
```

With [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script) & [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (recommended for reproducibility):

```bash
docker compose up --detach
docker compose exec r2flow bash
```

<details>
<summary>Without Docker:</summary>

```bash
ln -sf $KITTI360_ROOT ./r2flow/data/kitti_360/dataset
pip install -r requirements.txt
sudo apt install libsparsehash-dev # for torchsparse
pip install git+https://github.com/mit-han-lab/torchsparse.git@v2.0.0
pip install natten==0.17.1+torch210cu121 --find-links https://shi-labs.com/natten/wheels/
```
</details>

## Quick demo

Unconditional generation using the pre-trained model:

```python
import torch
import torchdiffeq

# Settings
model_config = "r2flow-kitti360-1rf"  # See https://github.com/kazuto1011/r2flow/releases/tag/weights
nfe = 256  # Number of function evaluations
batch_size = 1
device = "cuda"

# Setup a pre-trained model
model, lidar_utils, cfg = torch.hub.load(
    repo_or_dir="kazuto1011/r2flow", model="pretrained_r2flow", config=model_config, device=device
)

# Euler sampling
t = torch.linspace(0, 1, nfe + 1, device=device)
x0 = torch.randn(batch_size, model.in_channels, *model.resolution, device=device)
x1 = torchdiffeq.odeint(func=model, y0=x0, t=t, method="euler")[-1]

# Post-processing
range_image = lidar_utils.restore_metric_depth(x1[:, [0]])  # range in [0, 80]
rflct_image = lidar_utils.denormalize(x1[:, [1]])  # reflectance in [0, 1]
point_cloud = lidar_utils.convert_metric_depth(range_image, format="cartesian")
```

We also provide [a Gradio-based demo](https://huggingface.co/spaces/kazuto1011/r2flow) on Hugginface spaces.

## Training & evaluation

See [this page](docs/TRAINING.md) about how to reproduce R2Flow.

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{nakashima2025fast,
    title     = {Fast {LiDAR} Data Generation with Rectified Flows},
    author    = {Kazuto Nakashima and Xiaowen Liu and Tomoya Miyawaki and Yumi Iwashita and Ryo Kurazume},
    booktitle = {Proceedings of the International Conference on Robotics and Automation (ICRA)},
    pages     = {10057--10063},
    year      = 2025
}
```

## Acknowledgements

* The most part is based on our previous work ([`kazuto1011/r2dm`](https://github.com/kazuto1011/r2dm)).
* Rectified Flow is implemented using [`torchcfm`](https://github.com/atong01/conditional-flow-matching).
* HDiT is based on [`crowsonkb/k-diffusion`](https://github.com/crowsonkb/k-diffusion).
* JSD and MMD are based on [`vzyrianov/lidargen`](https://github.com/vzyrianov/lidargen).
* FRID, FSVD, and FPVD are based on [`hancyran/LiDAR-Diffusion`](https://github.com/hancyran/LiDAR-Diffusion)