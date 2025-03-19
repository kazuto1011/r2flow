from pathlib import Path

import torch

from r2flow.models.adm_unet import ADMUNet
from r2flow.models.efficient_unet import EfficientUNet
from r2flow.models.hdit import HDiT
from r2flow.utils.lidar import LiDARUtility
from r2flow.utils.option import DefaultConfig


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_model(
    ckpt: str | Path | dict,
    device: torch.device | str = "cpu",
    show_info: bool = True,
    compile: bool = False,
) -> tuple[torch.nn.Module, LiDARUtility, DefaultConfig]:
    if isinstance(ckpt, (str, Path)):
        ckpt = torch.load(ckpt, map_location="cpu")

    cfg = DefaultConfig(**ckpt["cfg"])

    channels = [
        3 if cfg.data.data_format == "cartesian" else 1,
        1 if cfg.data.train_reflectance else 0,
    ]

    if cfg.model.architecture == "efficient_unet":
        model = EfficientUNet(
            in_channels=sum(channels),
            resolution=cfg.data.resolution,
            base_channels=cfg.model.base_channels,
            temb_channels=cfg.model.temb_channels,
            channel_multiplier=cfg.model.channel_multiplier,
            num_residual_blocks=cfg.model.num_residual_blocks,
            gn_num_groups=cfg.model.gn_num_groups,
            gn_eps=cfg.model.gn_eps,
            attn_num_heads=cfg.model.attn_num_heads,
            coords_encoding=cfg.model.coords_encoding,
            ring=True,
        )
    elif cfg.model.architecture == "adm_unet":
        model = ADMUNet(
            in_channels=sum(channels),
            resolution=cfg.data.resolution,
            base_channels=cfg.model.base_channels,
            temb_channels=cfg.model.temb_channels,
            channel_multiplier=cfg.model.channel_multiplier,
            num_residual_blocks=cfg.model.num_residual_blocks,
            gn_num_groups=cfg.model.gn_num_groups,
            gn_eps=cfg.model.gn_eps,
            attn_num_heads=cfg.model.attn_num_heads,
            coords_encoding=cfg.model.coords_encoding,
            ring=True,
        )
    elif cfg.model.architecture == "nat_hdit":
        model = HDiT(
            in_channels=sum(channels),
            resolution=cfg.data.resolution,
            base_channels=cfg.model.base_channels,
            time_embed_channels=cfg.model.temb_channels,
            depths=cfg.model.num_residual_blocks,
            dilation=(1, 1, 1, 1),
            positional_embedding=cfg.model.coords_encoding,
            ring=True,
        )
    else:
        raise ValueError(f"Unknown: {cfg.model.architecture}")

    state_dict = ckpt["weights"]
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    if compile:
        model = torch.compile(model)

    lidar_utils = LiDARUtility(
        resolution=cfg.data.resolution,
        format=cfg.data.data_format,
        min_depth=cfg.data.min_depth,
        max_depth=cfg.data.max_depth,
        ray_angles=model.coords,
    )
    lidar_utils.eval()
    lidar_utils.to(device)

    if show_info:
        print(
            *[
                f"Resolution: {model.resolution}",
                f"Architecture: {model.__class__.__name__}",
                f"#images:  {ckpt['num_images']:,}",
                f"#params: {count_parameters(model):,}",
            ],
            sep="\n",
        )

    return model, lidar_utils, cfg


def setup_rng(seeds: list[int], device: torch.device | str):
    return [torch.Generator(device=device).manual_seed(i) for i in seeds]
