import dataclasses
import datetime
import json
import os
import warnings
from pathlib import Path

import accelerate
import datasets as ds
import einops
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
import torchcfm
import torchdiffeq
from ema_pytorch import EMA
from rich import print
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import r2flow.models
import r2flow.utils

warnings.filterwarnings("ignore", category=UserWarning)


def main(cfg: r2flow.utils.option.DefaultConfig):
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.suppress_errors = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # =================================================================================
    # Finetuning (optional, reflow/distillation)
    # =================================================================================

    is_finetuning = cfg.finetuning.init_ckpt is not None
    if is_finetuning:
        assert Path(cfg.finetuning.init_ckpt).exists()
        assert cfg.finetuning.sample_dir is not None
        model, lidar_utils, last_cfg = r2flow.utils.inference.setup_model(
            cfg.finetuning.init_ckpt
        )
        if cfg.finetuning.k_distil is not None:
            cfg.flow.num_sampling_steps = cfg.finetuning.k_distil
        r2flow.utils.training.show_diff(last_cfg, cfg)
    else:
        assert cfg.finetuning.k_distil is None, "k_distil should be None when 1-RF"

    # =================================================================================
    # Accelerator
    # =================================================================================

    project_dir = Path(cfg.training.output_dir) / cfg.data.dataset / cfg.data.projection

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=["tensorboard"],
        project_dir=project_dir,
        dynamo_backend=cfg.training.dynamo_backend,
        dataloader_config=accelerate.DataLoaderConfiguration(
            split_batches=True,
        ),
        step_scheduler_with_optimizer=True,
    )
    if accelerator.is_main_process:
        print(cfg)
        print(f"Number of processes (GPUs): {accelerator.num_processes}")
        os.makedirs(project_dir, exist_ok=True)
        project_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        accelerator.init_trackers(project_name=project_name)
        tracker = accelerator.get_tracker("tensorboard")
        path = Path(tracker.logging_dir) / "training_config.json"
        json.dump(dataclasses.asdict(cfg), open(path, "w"), indent=4)
    device = accelerator.device
    accelerate.utils.set_seed(cfg.training.seed, device_specific=True)

    num_steps_training = cfg.training.num_images_training // cfg.training.batch_size
    num_steps_lr_warmup = cfg.training.num_images_lr_warmup // cfg.training.batch_size

    # =================================================================================
    # Models
    # =================================================================================

    channels = [
        3 if cfg.data.data_format == "cartesian" else 1,
        1 if cfg.data.train_reflectance else 0,
    ]

    if not is_finetuning:
        if cfg.model.architecture == "efficient_unet":
            # Efficient U-Net [Saharia et al. NeurIPS 2022]
            # customized in R2DM [Nakashima et al. ICRA 2024]
            model = r2flow.models.EfficientUNet(
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
            # ADM U-Net [Dhariwal et al. NeurIPS 2021]
            model = r2flow.models.ADMUNet(
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
            # HDiT [Crowson et al. ICML 2024]
            # customized in R2Flow [Nakashima et al. ICRA 2025]
            model = r2flow.models.HDiT(
                in_channels=sum(channels),
                resolution=cfg.data.resolution,
                base_channels=cfg.model.base_channels,
                time_embed_channels=cfg.model.temb_channels,
                depths=cfg.model.num_residual_blocks,
                num_heads=cfg.model.attn_num_heads,
                dilation=(1, 1, 1, 1),
                positional_embedding=cfg.model.coords_encoding,
                ring=True,
            )
        else:
            raise ValueError(f"Unknown: {cfg.model.architecture}")

    if "spherical" in cfg.data.projection:
        # Spherical projection
        model.coords = r2flow.utils.lidar.get_hdl64e_linear_ray_angles(
            *cfg.data.resolution
        )
    elif "unfolding" in cfg.data.projection:
        # Scan unfolding
        model.coords = F.interpolate(
            torch.load(f"data/{cfg.data.dataset}/unfolding_angles.pth"),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
    else:
        raise ValueError(f"Unknown: {cfg.data.projection}")

    if accelerator.is_main_process:
        print(
            f"Number of model parameters: {r2flow.utils.inference.count_parameters(model):,}"
        )

    if accelerator.is_main_process:
        ema = EMA(
            model,
            beta=cfg.training.ema_decay,
            update_every=cfg.training.ema_update_every,
            update_after_step=num_steps_lr_warmup
            * cfg.training.gradient_accumulation_steps,
        )
        ema.to(device)

    if not is_finetuning:
        lidar_utils = r2flow.utils.lidar.LiDARUtility(
            resolution=cfg.data.resolution,
            format=cfg.data.data_format,
            min_depth=cfg.data.min_depth,
            max_depth=cfg.data.max_depth,
            ray_angles=model.coords,
        )
        lidar_utils.to(device)

    # =================================================================================
    # Flow matching
    # =================================================================================

    if cfg.flow.formulation == "icfm":
        # This is equivalent to 1-rectified flow if sigma is 0
        fm = torchcfm.conditional_flow_matching.ConditionalFlowMatcher(
            sigma=cfg.flow.sigma
        )
    elif cfg.flow.formulation == "otcfm":
        # No improvement was observed with OT couplings, which may be due to a small batch size.
        fm = torchcfm.conditional_flow_matching.ExactOptimalTransportConditionalFlowMatcher(
            sigma=cfg.flow.sigma
        )
    else:
        raise ValueError(f"Unknown: {cfg.flow.formulation}")

    timestep_sampler = r2flow.utils.timestep_sampler.TimestepSampler(
        distribution=cfg.training.timestep_distribution,
        k_distill=cfg.finetuning.k_distil,
    )

    if cfg.training.loss_fn == "l2":
        # Rectified Flow [Liu et al. ICLR 2023]
        loss_fn = F.mse_loss
    elif cfg.training.loss_fn == "l1":
        loss_fn = F.l1_loss
    elif cfg.training.loss_fn == "smooth_l1":
        loss_fn = F.smooth_l1_loss
    elif cfg.training.loss_fn == "pseudo_huber":
        # 2-Rectifed Flow++ [Lee et al. NeurIPS 2024]
        loss_fn = r2flow.utils.training.pseudo_huber_loss
    else:
        raise ValueError(f"Unknown: {cfg.training.loss_fn}")

    # =================================================================================
    # Optimizer & dataloader
    # =================================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        weight_decay=cfg.training.adam_weight_decay,
        eps=cfg.training.adam_epsilon,
    )

    if is_finetuning:
        dataset = r2flow.utils.training.IndexedSamples(
            sample_dir=cfg.finetuning.sample_dir,
        )
    else:
        dataset = ds.load_dataset(
            path=f"r2flow/data/{cfg.data.dataset}",
            name=cfg.data.projection,
            split=ds.Split.TRAIN,
            num_proc=cfg.training.num_workers,
            trust_remote_code=True,
        ).with_format("torch")

    if accelerator.is_main_process:
        print(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    lr_scheduler = r2flow.utils.training.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_steps_lr_warmup * cfg.training.gradient_accumulation_steps,
        num_training_steps=num_steps_training
        * cfg.training.gradient_accumulation_steps,
    )

    model, lidar_utils, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, lidar_utils, optimizer, dataloader, lr_scheduler
    )

    # =================================================================================
    # Utilities
    # =================================================================================

    def preprocess(batch):
        x = []
        if cfg.data.data_format == "cartesian":
            mask = lidar_utils.get_mask(batch["depth"])
            x += [batch["xyz"] / lidar_utils.max_depth * mask]
        else:
            x += [lidar_utils.convert_metric_depth(batch["depth"])]
        if cfg.data.train_reflectance:
            x += [lidar_utils.normalize(batch["reflectance"])]
        x = torch.cat(x, dim=1)
        x = F.interpolate(x.to(device), size=cfg.data.resolution, mode="nearest-exact")
        return x

    def split_channels(image: torch.Tensor):
        range_image, rflct_image = torch.split(image, channels, dim=1)
        return range_image, rflct_image

    @torch.inference_mode()
    def log_images(image, tag: str = "name", global_step: int = 0):
        out = dict()
        range_image, rflct_image = split_channels(image)
        metric_depth = lidar_utils.restore_metric_depth(range_image)
        out[f"{tag}/depth"] = r2flow.utils.render.colorize(
            metric_depth / lidar_utils.max_depth
        )

        if cfg.data.data_format == "cartesian":
            xyz = range_image
        else:
            xyz = lidar_utils.convert_metric_depth(metric_depth, format="cartesian")
            xyz = xyz / lidar_utils.max_depth
        normal = -r2flow.utils.render.estimate_surface_normal(xyz)
        normal = lidar_utils.denormalize(normal)
        bev = r2flow.utils.render.render_point_clouds(
            points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
            colors=einops.rearrange(normal, "B C H W -> B (H W) C"),
            t=torch.tensor([0, 0, 1.0]).to(xyz),
        )
        out[f"{tag}/bev"] = bev.mul(255).clamp(0, 255).byte()

        if rflct_image.numel() > 0:
            rflct_image = lidar_utils.denormalize(rflct_image)
            out[f"{tag}/rflct"] = r2flow.utils.render.colorize(rflct_image, cm.plasma)

        mask = lidar_utils.get_mask(metric_depth)
        out[f"{tag}/mask"] = r2flow.utils.render.colorize(mask, cm.binary_r)

        tracker.log_images(out, step=global_step)

    def sync(fn, *args, **kwargs):
        # This is for computing OT over all processes
        total_rank = accelerator.num_processes
        rank = accelerator.process_index
        args, kwargs = accelerator.gather((args, kwargs))
        outputs = fn(*args, **kwargs)
        outputs = accelerate.utils.broadcast(outputs)
        return [o.chunk(total_rank)[rank] for o in outputs]

    # =================================================================================
    # Sampling
    # =================================================================================

    ode_kwargs = dict(
        y0=torch.randn(
            cfg.training.batch_size // accelerator.num_processes,
            sum(channels),
            *cfg.data.resolution,
            device=device,
        ),
        t=torch.linspace(0, 1, cfg.flow.num_sampling_steps + 1, device=device),
        method="dopri5",
        atol=1e-5,
        rtol=1e-5,
    )

    # =================================================================================
    # Main loop
    # =================================================================================

    progress_bar = tqdm(
        range(num_steps_training),
        desc="training",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    global_step = 0
    while global_step < num_steps_training:
        model.train()
        for batch in dataloader:
            # =================================================================================
            # Training
            # =================================================================================

            if is_finetuning:
                x_1 = preprocess(batch[1])
                x_0 = r2flow.utils.training.restore_x_0(batch[0], x_1.shape[1:], device)
            else:
                x_1 = preprocess(batch)
                x_0 = torch.randn_like(x_1)

            t, x_t, u_t = sync(
                fm.sample_location_and_conditional_flow,
                x0=x_0,
                x1=x_1,
                t=timestep_sampler(x_1.shape[0], device=device),
            )

            with accelerator.accumulate(model):
                v_t = model(t=t, x=x_t)
                loss = loss_fn(v_t, u_t)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # =================================================================================
            # Logging
            # =================================================================================

            global_step += 1
            num_images = global_step * cfg.training.batch_size
            log = {f"training/loss/{cfg.training.loss_fn}": loss.item()}
            for j, group in enumerate(optimizer.param_groups):
                log[f"training/lr/{j}"] = group["lr"]

            if accelerator.is_main_process:
                ema.update()
                log["training/ema/decay"] = ema.get_current_decay()

                if global_step == 1:
                    log_images(x_1, "dataset", num_images)

                if global_step % cfg.training.num_steps_logging == 0:
                    ema.ema_model.nfe = 0
                    with torch.no_grad():
                        x_1_ = torchdiffeq.odeint(func=ema.ema_model, **ode_kwargs)[-1]
                    log["validation/nfe"] = ema.ema_model.nfe
                    log_images(x_1_, "sample", num_images)

                if global_step % cfg.training.num_steps_checkpoint == 0:
                    save_dir = Path(tracker.logging_dir) / "models"
                    save_dir.mkdir(exist_ok=True, parents=True)
                    torch.save(
                        {
                            "cfg": dataclasses.asdict(cfg),
                            "weights": ema.ema_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "num_images": num_images,
                        },
                        save_dir / f"checkpoint_{num_images:010d}.pth",
                    )

            accelerator.log(log, step=num_images)
            progress_bar.update(1)
            if global_step >= num_steps_training:
                break

    accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(r2flow.utils.option.DefaultConfig, dest="cfg")
    main(parser.parse_args().cfg)
