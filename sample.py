import warnings
from argparse import ArgumentParser
from pathlib import Path

import accelerate
import natten
import torch
import torchdiffeq
from joblib import Parallel, delayed
from rich import print
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq._impl.odeint import SOLVERS
from tqdm.auto import tqdm

import r2flow.utils

warnings.filterwarnings("ignore", category=UserWarning)


def filter_seeds(seeds_full, args):
    index = torch.ones_like(seeds_full)
    for seed in tqdm(seeds_full, desc="Removing existing seeds...", leave=False):
        if (Path(args.output_dir) / f"coupling_{seed:010d}.pth").exists():
            index[seed] = 0
    return index


def sample(args):
    torch.set_grad_enabled(False)
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.suppress_errors = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    natten.use_fused_na(True)

    # =================================================================================
    # Loading model
    # =================================================================================

    model, lidar_utils, cfg = r2flow.utils.inference.setup_model(args.ckpt)
    shape = (model.in_channels, *cfg.data.resolution)
    channels = [
        3 if cfg.data.data_format == "cartesian" else 1,
        1 if cfg.data.train_reflectance else 0,
    ]

    # =================================================================================
    # Accelerator
    # =================================================================================

    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.training.mixed_precision,
        dynamo_backend=cfg.training.dynamo_backend,
        dataloader_config=accelerate.DataLoaderConfiguration(
            split_batches=True,
            even_batches=False,
        ),
        step_scheduler_with_optimizer=True,
    )
    device = accelerator.device
    accelerate.utils.set_seed(cfg.training.seed, device_specific=True)

    if accelerator.is_main_process:
        print(cfg)

    output_dir = Path(args.output_dir)
    with accelerator.main_process_first():
        output_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================================
    # Sampling seeds
    # =================================================================================

    seeds_full = torch.arange(args.num_samples, device=device).long()
    if accelerator.is_main_process:
        index = filter_seeds(seeds_full, args).to(device)
    else:
        index = torch.ones_like(seeds_full).to(device)
    accelerator.wait_for_everyone()
    accelerate.utils.broadcast(index, from_process=0)
    z_0_seeds = seeds_full[index == 1].cpu()

    if accelerator.is_main_process:
        print(f"#seeds: {len(z_0_seeds):,}/{len(seeds_full):,}")

    dataloader = DataLoader(
        TensorDataset(z_0_seeds),
        batch_size=args.batch_size,
        num_workers=cfg.training.num_workers,
        drop_last=False,
    )

    model.to(device)
    model, lidar_utils, dataloader = accelerator.prepare(model, lidar_utils, dataloader)

    # =================================================================================
    # Utilities
    # =================================================================================

    def sample_z_0(seeds):
        z_0_list = []
        for seed in seeds.cpu().tolist():
            rng = torch.Generator(device=device).manual_seed(seed)
            z_0_list.append(torch.randn(*shape, device=device, generator=rng))
        return torch.stack(z_0_list, dim=0)

    @torch.inference_mode()
    def sample_z_1(z_0):
        return torchdiffeq.odeint(
            func=model,
            y0=z_0,
            t=torch.linspace(0, 1, args.num_steps + 1, device=device),
            method=args.solver,
            atol=args.tol,
            rtol=args.tol,
            options=dict(min_step=1e-3),
        )[-1]

    def postprocess(samples):
        samples = samples.clamp(-1, 1)
        depth, rflct = torch.split(samples, channels, dim=1)
        metric_depth = lidar_utils.restore_metric_depth(depth)
        rflct = lidar_utils.denormalize(rflct)
        if cfg.data.data_format == "cartesian":
            xyz = depth * lidar_utils.max_depth
            xyz = xyz * lidar_utils.get_mask(metric_depth)
        else:
            xyz = lidar_utils.convert_metric_depth(metric_depth, format="cartesian")
        return (
            metric_depth.float().cpu(),
            xyz.float().cpu(),
            rflct.float().cpu(),
        )

    # =================================================================================
    # Sampling
    # =================================================================================

    progress = tqdm(
        dataloader,
        desc="Generating...",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    )

    for z_0_seeds in progress:
        if z_0_seeds is None:
            break
        else:
            (z_0_seeds,) = z_0_seeds

        z_0 = sample_z_0(z_0_seeds)
        z_1 = sample_z_1(z_0)

        depth_cpu, xyz_cpu, rflct_cpu = postprocess(z_1)
        Parallel(n_jobs=cfg.training.num_workers)(
            delayed(torch.save)(
                {
                    "depth": depth_cpu[i].clone(),
                    "xyz": xyz_cpu[i].clone(),
                    "reflectance": rflct_cpu[i].clone(),
                },
                Path(args.output_dir) / f"sample_{seed:010d}.pth",
            )
            for i, seed in enumerate(z_0_seeds)
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--solver", choices=SOLVERS.keys(), default="euler")
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()
    sample(args)
