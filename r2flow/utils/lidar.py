from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def get_hdl64e_linear_ray_angles(
    H: int = 64, W: int = 2048, device: torch.device = "cpu"
):
    h_up, h_down = 3, -25
    w_left, w_right = 180, -180
    elevation = 1 - torch.arange(H, device=device) / H  # [0, 1]
    elevation = elevation * (h_up - h_down) + h_down  # [-25, 3]
    azimuth = 1 - torch.arange(W, device=device) / W  # [0, 1]
    azimuth = azimuth * (w_left - w_right) + w_right  # [-180, 180]
    [elevation, azimuth] = torch.meshgrid([elevation, azimuth], indexing="ij")
    angles = torch.stack([elevation, azimuth])[None].deg2rad()
    return angles


class LiDARUtility(nn.Module):
    def __init__(
        self,
        resolution: tuple[int, int],
        format: Literal["logscale", "inverse", "metric", "cartesian"],
        min_depth: float,
        max_depth: float,
        ray_angles: torch.Tensor = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.format = format
        self.min_depth = min_depth
        self.max_depth = max_depth
        if ray_angles is None:
            ray_angles = get_hdl64e_linear_ray_angles(*resolution)
        else:
            assert ray_angles.ndim == 4 and ray_angles.shape[1] == 2
        ray_angles = F.interpolate(
            ray_angles,
            size=self.resolution,
            mode="nearest-exact",
        )
        self.register_buffer("ray_angles", ray_angles.float())

    @staticmethod
    def denormalize(x: torch.Tensor) -> torch.Tensor:
        """Scale from [-1, +1] to [0, 1]"""
        return ((x + 1) / 2).clamp(0, 1)

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        """Scale from [0, 1] to [-1, +1]"""
        return (x * 2 - 1).clamp(-1, 1)

    def get_mask(self, metric):
        mask = (metric > self.min_depth) & (metric < self.max_depth)
        return mask.float()

    @torch.no_grad()
    def convert_metric_depth(
        self,
        metric_depth: torch.Tensor,
        mask: torch.Tensor | None = None,
        format: str = None,
    ) -> torch.Tensor:
        """
        Convert metric depth in [0, `max_depth`] to normalized depth in [-1, 1].
        """
        if format is None:
            format = self.format
        if mask is None:
            mask = self.get_mask(metric_depth)
        if format == "logscale":
            converted_depth = torch.log2(metric_depth + 1) / np.log2(self.max_depth + 1)
            converted_depth = self.normalize(converted_depth * mask)
        elif format == "inverse":
            converted_depth = self.min_depth / metric_depth.add(1e-8)
            converted_depth = self.normalize(converted_depth * mask)
        elif format == "metric":
            converted_depth = metric_depth.div(self.max_depth)
            converted_depth = self.normalize(converted_depth * mask)
        elif format == "cartesian":
            """metric -> xyz is irreversible if spherical projection"""
            phi = self.ray_angles[:, [0]]
            theta = self.ray_angles[:, [1]]
            grid_x = metric_depth * phi.cos() * theta.cos()
            grid_y = metric_depth * phi.cos() * theta.sin()
            grid_z = metric_depth * phi.sin()
            converted_depth = torch.cat((grid_x, grid_y, grid_z), dim=1) * mask
        else:
            raise ValueError("Invalid depth format")
        return converted_depth

    @torch.no_grad()
    def restore_metric_depth(
        self,
        converted_depth: torch.Tensor,
        format: str = None,
    ) -> torch.Tensor:
        """
        Revert normalized depth in [-1, 1] back to metric depth in [0, `max_depth`].
        """
        if format is None:
            format = self.format
        if format == "logscale":
            converted_depth = self.denormalize(converted_depth)
            metric_depth = torch.exp2(converted_depth * np.log2(self.max_depth + 1)) - 1
        elif format == "inverse":
            converted_depth = self.denormalize(converted_depth)
            metric_depth = self.min_depth / converted_depth.add(1e-8)
        elif format == "metric":
            converted_depth = self.denormalize(converted_depth)
            metric_depth = converted_depth.mul(self.max_depth)
        elif format == "cartesian":
            converted_depth = converted_depth * self.max_depth
            metric_depth = torch.norm(converted_depth, dim=1, p=2, keepdim=True)
        else:
            raise ValueError
        return metric_depth * self.get_mask(metric_depth)
