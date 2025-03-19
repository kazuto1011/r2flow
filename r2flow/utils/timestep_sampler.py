import math
from typing import Literal

import torch


def u_shaped(num_samples, const=4, device="cpu"):
    """
    Improving the Training of Rectified Flows
    https://arxiv.org/abs/2405.20320
    """
    C = const / (math.exp(const) - 1)
    u = torch.rand(num_samples, device=device)
    t = torch.log(u * const / C + 1) / const
    t = torch.cat([t, 1 - t], dim=0)
    t = t[torch.randperm(t.shape[0])[:num_samples]]
    return t


def logit_normal(num_samples, mean=0.0, std=1.0, device="cpu"):
    """
    Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (Stable Diffusion 3)
    https://arxiv.org/abs/2403.03206
    """
    n = torch.normal(mean, std, (num_samples,), device=device)
    t = torch.sigmoid(n)
    return t


class TimestepSampler:
    def __init__(
        self,
        distribution: Literal[
            "uniform", "u_shaped", "logit_normal", "zeros"
        ] = "uniform",
        k_distill: int | None = None,
        **kwargs,
    ):
        self.distribution = distribution
        self.k_distill = k_distill
        self.kwargs = kwargs

    def __call__(self, num_samples, device="cpu"):
        if self.distribution == "uniform":
            t = torch.rand(num_samples, device=device)
        elif self.distribution == "u_shaped":
            t = u_shaped(num_samples, **self.kwargs, device=device)
        elif self.distribution == "logit_normal":
            t = logit_normal(num_samples, **self.kwargs, device=device)
        else:
            raise ValueError(f"Unknown timestep distribution: {self.distribution}")
        if self.k_distill is not None:
            t = torch.floor(t * self.k_distill) / self.k_distill
        return t
