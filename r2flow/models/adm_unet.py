# =============================================================================
# Reimplementation of the ADM-UNet:
# https://arxiv.org/abs/2105.05233
# =============================================================================

from typing import Iterable, Literal

import einops
import numpy as np
import torch
from torch import nn

from . import encoding, ops


def _join(*tensors) -> torch.Tensor:
    return torch.cat(tensors, dim=1)


def _n_tuple(x: Iterable | int, N: int) -> tuple[int]:
    if isinstance(x, Iterable):
        assert len(x) == N
        return x
    else:
        return (x,) * N


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        gn_eps: float = 1e-6,
        gn_num_groups: int = 8,
        scale: float = 1 / np.sqrt(2),
    ):
        super().__init__()
        self.norm = nn.GroupNorm(gn_num_groups, in_channels, gn_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn.out_proj.apply(ops.zero_out)
        self.register_buffer("scale", torch.tensor(scale).float())

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        B, C, H, W = h.shape
        h = einops.rearrange(h, "B C H W -> B (H W) C")
        h, _ = self.attn(query=h, key=h, value=h, need_weights=False)
        h = einops.rearrange(h, "B (H W) C -> B C H W", H=H, W=W)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.residual(x)
        h = h * self.scale
        return h


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int | None,
        gn_num_groups: int = 8,
        gn_eps: float = 1e-6,
        scale: float = 1 / np.sqrt(2),
        dropout: float = 0.0,
        resample: Literal["up", "down", None] = None,
        attn: bool = False,
        attn_num_heads: int = 8,
        ring: bool = False,
    ):
        super().__init__()
        self.has_emb = emb_channels is not None

        if resample == "up":
            self.resample = ops.Resample(up=2, window=[1, 1], ring=ring)
        elif resample == "down":
            self.resample = ops.Resample(down=2, window=[1, 1], ring=ring)
        else:
            self.resample = nn.Identity()

        # layer 1
        self.norm1 = nn.GroupNorm(gn_num_groups, in_channels, gn_eps)
        self.silu1 = nn.SiLU()
        self.conv1 = ops.Conv2d(in_channels, out_channels, 3, 1, 1, ring=ring)

        # layer 2
        if self.has_emb:
            self.norm2 = ops.AdaGN(emb_channels, out_channels, gn_num_groups, gn_eps)
        else:
            self.norm2 = nn.GroupNorm(gn_num_groups, out_channels, gn_eps)
        self.silu2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)
        self.conv2 = ops.Conv2d(out_channels, out_channels, 3, 1, 1, ring=ring)
        self.conv2.apply(ops.zero_out)

        # skip connection
        self.skip_conv = (
            ops.Conv2d(in_channels, out_channels, 1, 1, 0)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.register_buffer("scale", torch.tensor(scale).float())

        self.self_attn_block = (
            SelfAttentionBlock(
                in_channels=out_channels,
                num_heads=attn_num_heads,
                gn_eps=gn_eps,
                gn_num_groups=gn_num_groups,
            )
            if attn
            else nn.Identity()
        )

    def residual(
        self, x: torch.Tensor, emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.resample(h)
        h = self.silu1(h)
        h = self.conv1(h)
        h = self.norm2(h, emb) if self.has_emb else self.norm2(h)
        h = self.silu2(h)
        h = self.drop2(h)
        h = self.conv2(h)
        return h

    def skip(self, x: torch.Tensor) -> torch.Tensor:
        h = self.skip_conv(x)
        h = self.resample(h)
        return h

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.skip(x) + self.residual(x, emb)
        h = h * self.scale
        h = self.self_attn_block(h)
        return h


class ADMUNet(nn.Module):
    """
    Diffusion Models Beat GANs on Image Synthesis, NeurIPS'21
    """

    def __init__(
        self,
        in_channels: int,
        resolution: tuple[int, int] | int,
        out_channels: int | None = None,  # == in_channels if None
        base_channels: int = 128,
        temb_channels: int = None,
        channel_multiplier: tuple[int] | int = (1, 2, 4, 8),
        num_residual_blocks: tuple[int] | int = (3, 3, 3, 3),
        attention_levels: tuple[int] = (3,),
        gn_num_groups: int = 32 // 4,
        gn_eps: float = 1e-6,
        attn_num_heads: int = 8,
        coords_encoding: Literal[
            "spherical_harmonics",
            "polar_coordinates",
            "fourier_features",
            None,
        ] = "spherical_harmonics",
        ring: bool = True,
    ):
        super().__init__()
        self.resolution = _n_tuple(resolution, 2)
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        temb_channels = base_channels * 4 if temb_channels is None else temb_channels

        # spatial coords embedding
        coords = encoding.generate_polar_coords(*self.resolution)
        self.register_buffer("coords", coords)
        self.coords_encoding = None
        if coords_encoding == "spherical_harmonics":
            self.coords_encoding = encoding.SphericalHarmonics(levels=5)
            in_channels += self.coords_encoding.extra_ch
        elif coords_encoding == "polar_coordinates":
            self.coords_encoding = nn.Identity()
            in_channels += coords.shape[1]
        elif coords_encoding == "fourier_features":
            self.coords_encoding = encoding.FourierFeatures(self.resolution)
            in_channels += self.coords_encoding.extra_ch

        # timestep embedding
        self.time_embedding = nn.Sequential(
            ops.SinusoidalPositionalEmbedding(base_channels),
            nn.Linear(base_channels, temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels),
        )

        # parameters for up/down-sampling blocks
        resolution_levels = 4
        channel_multiplier = _n_tuple(channel_multiplier, resolution_levels)
        num_residual_blocks = _n_tuple(num_residual_blocks, resolution_levels)

        block_cfgs = dict(
            emb_channels=temb_channels,
            gn_num_groups=gn_num_groups,
            gn_eps=gn_eps,
            attn_num_heads=attn_num_heads,
            dropout=0.0,
            ring=ring,
        )

        channels = base_channels * channel_multiplier[0]
        self.in_conv = ops.Conv2d(in_channels, channels, 3, 1, 1, ring=ring)
        skip_channels = [channels]

        # downsampling blocks
        self.down_blocks = nn.ModuleList()
        for level in range(resolution_levels):
            is_final_level = level == resolution_levels - 1
            channels = base_channels * channel_multiplier[level]
            for i in range(num_residual_blocks[level]):
                self.down_blocks.append(
                    ResidualBlock(
                        in_channels=skip_channels[-1],
                        out_channels=channels,
                        resample=None,
                        attn=level in attention_levels,
                        **block_cfgs,
                    )
                )
                skip_channels.append(channels)
            if not is_final_level:
                self.down_blocks.append(
                    ResidualBlock(
                        in_channels=channels,
                        out_channels=channels,
                        resample="down",
                        attn=False,
                        **block_cfgs,
                    )
                )
                skip_channels.append(channels)

        # middle blocks
        self.mid_blocks = ops.ConditionalSequential(
            ResidualBlock(
                in_channels=skip_channels[-1],
                out_channels=skip_channels[-1],
                attn=True,
                **block_cfgs,
            ),
            ResidualBlock(
                in_channels=skip_channels[-1],
                out_channels=skip_channels[-1],
                attn=False,
                **block_cfgs,
            ),
        )

        # upsampling blocks
        last_channels = skip_channels[-1]
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(resolution_levels)):
            is_final_level = level == 0
            channels = base_channels * channel_multiplier[level]
            for i in range(num_residual_blocks[level] + 1):
                is_first_block = i == 0
                is_final_block = i == num_residual_blocks[level]
                block = ops.ConditionalSequential()
                block.append(
                    ResidualBlock(
                        in_channels=(last_channels if is_first_block else channels)
                        + skip_channels.pop(),
                        out_channels=channels,
                        attn=level in attention_levels,
                        **block_cfgs,
                    )
                )
                if not is_final_level and is_final_block:
                    block.append(
                        ResidualBlock(
                            in_channels=channels,
                            out_channels=channels,
                            resample="up",
                            attn=False,
                            **block_cfgs,
                        )
                    )
                last_channels = channels
                self.up_blocks.append(block)

        channels = base_channels * channel_multiplier[0]
        self.out_conv = nn.Sequential(
            nn.GroupNorm(gn_num_groups, channels, gn_eps),
            nn.SiLU(),
            ops.Conv2d(channels, self.out_channels, 3, 1, 1, ring=ring),
        )
        self.out_conv[-1].apply(ops.zero_out)

    def forward(
        self, t: torch.Tensor, x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        h = x

        # timestep embedding
        if len(t.shape) == 0:
            t = t[None].repeat_interleave(h.shape[0], dim=0)
        temb = self.time_embedding(t.to(h))

        # spatial embedding
        if self.coords_encoding is not None:
            cenc = self.coords_encoding(self.coords)
            cenc = cenc.repeat_interleave(h.shape[0], dim=0)
            h = torch.cat([h, cenc], dim=1)

        # u-net part
        skip = [self.in_conv(h)]
        for block in self.down_blocks:
            skip.append(block(skip[-1], temb))
        h = self.mid_blocks(skip[-1], temb)
        for block in self.up_blocks:
            h = block(_join(h, skip.pop()), temb)
        h = self.out_conv(h)

        return h
