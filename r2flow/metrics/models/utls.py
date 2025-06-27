import io
import os
import sys
import tarfile
from itertools import repeat
from types import SimpleNamespace
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
import torchsparse
import yaml
from torch import nn
from torchsparse.utils.collate import sparse_collate


def ravel_hash(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, x.shape
    x = (x - x.amin(dim=0)).long()
    xmax = x.amax(dim=0).long() + 1
    h = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h


def unique(x, dim=0):
    # https://github.com/pytorch/pytorch/issues/36748#issuecomment-1454524266
    unique, inverse, counts = torch.unique(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    decimals = torch.arange(inverse.numel(), device=inverse.device) / inverse.numel()
    inv_sorted = (inverse + decimals).argsort()
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    index = index.sort().values
    return unique, inverse, counts, index


def sparse_quantize(coords, voxel_size=1):
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, coords.shape[1]))
    assert isinstance(voxel_size, tuple) and len(voxel_size) in [2, 3]
    voxel_size = torch.tensor(voxel_size).to(coords)
    coords = torch.floor(coords / voxel_size).int()
    _, inv_inds, _, inds = unique(ravel_hash(coords))
    return coords[inds], inds, inv_inds


class PreProcess(nn.Module):
    def __init__(self, voxel_size=0.05):
        super().__init__()
        self.voxel_size = voxel_size

    def voxelize(self, pcd):
        pcd = pcd[pcd.norm(p=2, dim=1) > 0]
        coords = (pcd / self.voxel_size).round()
        coords = coords - coords.amin(dim=0)
        feat = F.pad(pcd, pad=(0, 1), value=-1)
        # remove duplicates
        _, inds, inv_inds = sparse_quantize(coords, voxel_size=1)
        feat = feat[inds].float()
        coords = coords[inds].long()
        voxel = torchsparse.SparseTensor(feat, coords)
        return voxel

    def forward(self, pcds):
        voxel = sparse_collate([self.voxelize(pcd) for pcd in pcds])
        return {"lidar": voxel}


class PostProcess(nn.Module):
    def __init__(
        self,
        num_sectors: int = 16,
        voxel_size: float = 0.05,
        min_depth: float = 1.0,
        max_depth: float = 56.0,
    ):
        super().__init__()
        self.num_sectors = num_sectors
        self.voxel_size = voxel_size
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, input_dict, agg_type: str = "depth"):
        output_list = []
        batch_indices = input_dict["batch_indices"]
        for b_idx in range(batch_indices.max() + 1):
            if agg_type == "global":
                feats = input_dict["features"][batch_indices == b_idx].mean(dim=0)
            elif agg_type == "angle":
                feats = input_dict["features"][batch_indices == b_idx]
                coords = input_dict["coords"][batch_indices == b_idx].float()
                coords = coords - coords.mean(dim=0)
                angle = torch.atan2(coords[:, 1], coords[:, 0])  # [-pi, pi]
                sector_range = torch.linspace(
                    -np.pi - 1e-4, np.pi + 1e-4, self.num_sectors + 1
                )
                feats_list = []
                for s_idx in range(self.num_sectors):
                    sector_indices = torch.where(
                        (angle >= sector_range[s_idx])
                        & (angle < sector_range[s_idx + 1])
                    )[0]
                    sector_feats = feats[sector_indices].mean(dim=0)
                    sector_feats = torch.nan_to_num(sector_feats, 0.0)
                    feats_list.append(sector_feats)
                feats = torch.cat(feats_list)  # dim: 768
            elif agg_type == "depth":
                feats = input_dict["features"][batch_indices == b_idx]
                coords = input_dict["coords"][batch_indices == b_idx].float()
                coords = coords - coords.mean(dim=0)
                bev_depth = torch.norm(coords, dim=-1) * self.voxel_size
                sector_range = torch.linspace(
                    self.min_depth + 3, self.max_depth, self.num_sectors + 1
                )
                sector_range[0] = 0.0
                feats_list = []
                for s_idx in range(self.num_sectors):
                    sector_indices = torch.where(
                        (bev_depth >= sector_range[s_idx])
                        & (bev_depth < sector_range[s_idx + 1])
                    )[0]
                    sector_feats = feats[sector_indices].mean(dim=0)
                    sector_feats = torch.nan_to_num(sector_feats, 0.0)
                    feats_list.append(sector_feats)
                feats = torch.cat(feats_list)  # dim: 768
            elif agg_type is None:
                feats = input_dict["features"][batch_indices == b_idx]
            else:
                raise NotImplementedError
            output_list.append(feats)
        if agg_type is not None:
            output_list = torch.stack(output_list)
        return output_list


def dict_to_namespace(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)


def setup_voxel_models_from_lidm(
    url_or_file: str,
    model_class: torch.nn.Module,
    progress: bool = True,
    compile: bool = False,
    voxel_size: float = 0.05,
    num_sectors: int = 16,
    min_depth: float = 1.0,
    max_depth: float = 56.0,
):
    # set the cache directory
    hub_dir = torch.hub.get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)

    # download the tar file
    parts = urlparse(url_or_file)
    filename = os.path.basename(parts.path)
    arch = filename.replace(".tar.gz", "")
    if all([parts.scheme, parts.netloc]):
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            sys.stderr.write(
                'Downloading: "{}" to {}\n'.format(url_or_file, cached_file)
            )
            hash_prefix = None
            torch.hub.download_url_to_file(
                url_or_file, cached_file, hash_prefix, progress=progress
            )
    else:
        cached_file = url_or_file

    # parse the downloaded tar file
    with tarfile.open(cached_file, "r:gz") as tar:
        stream = io.BytesIO(tar.extractfile(f"{arch}/model.ckpt").read())
        ckpt = torch.load(stream, map_location="cpu")
        stream = io.BytesIO(tar.extractfile(f"{arch}/config.yaml").read())
        config = dict_to_namespace(yaml.safe_load(stream))

    # setup the model
    model = model_class(config)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval().requires_grad_(False)
    if compile:
        model = torch.compile(model)

    return (
        model,
        PreProcess(voxel_size=voxel_size),
        PostProcess(
            voxel_size=voxel_size,
            num_sectors=num_sectors,
            min_depth=min_depth,
            max_depth=max_depth,
        ),
    )


# =================================================================================
# Visualization utilities
# =================================================================================

_ID2LABEL = {
    0: "unlabeled",
    1: "car",
    2: "bicycle",
    3: "motorcycle",
    4: "truck",
    5: "other-vehicle",
    6: "person",
    7: "bicyclist",
    8: "motorcyclist",
    9: "road",
    10: "parking",
    11: "sidewalk",
    12: "other-ground",
    13: "building",
    14: "fence",
    15: "vegetation",
    16: "trunk",
    17: "terrain",
    18: "pole",
    19: "traffic-sign",
}


def make_semantickitti_cmap():
    from matplotlib.colors import ListedColormap

    label_colors = {
        0: [0, 0, 0],
        1: [245, 150, 100],
        2: [245, 230, 100],
        3: [150, 60, 30],
        4: [180, 30, 80],
        5: [255, 0, 0],
        6: [30, 30, 255],
        7: [200, 40, 255],
        8: [90, 30, 150],
        9: [255, 0, 255],
        10: [255, 150, 255],
        11: [75, 0, 75],
        12: [75, 0, 175],
        13: [0, 200, 255],
        14: [50, 120, 255],
        15: [0, 175, 0],
        16: [0, 60, 135],
        17: [80, 240, 150],
        18: [150, 240, 255],
        19: [0, 0, 255],
    }
    num_classes = max(label_colors.keys()) + 1
    label_colormap = np.zeros((num_classes, 3), dtype=np.uint8)
    for label_id, color in label_colors.items():
        label_colormap[label_id] = color[::-1]  # BGR -> RGB
    cmap = ListedColormap(label_colormap / 255.0)
    return cmap
