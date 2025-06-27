import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import datasets as ds
import einops
import natten
import torch
import torch.nn.functional as F
from rich import print
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import r2flow.metrics
import r2flow.utils

# from LiDARGen
MAX_DEPTH = 63.0
MIN_DEPTH = 0.5
KITTI_MAX_DEPTH = 80.0


def resize(x, size):
    return F.interpolate(x, size=size, mode="nearest-exact")


@torch.no_grad()
def evaluate(args):
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

    cfg = r2flow.utils.option.DefaultConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W = cfg.data.resolution
    proj = r2flow.metrics.extractor.FeatureExtractor((H, W), compile=args.compile)
    proj.eval()
    proj.to(device)

    print(f"Metrics: {proj.available_metrics()}")

    # =====================================================
    # real set
    # =====================================================

    loader_real = DataLoader(
        dataset=ds.load_dataset(
            path=f"r2flow/data/{cfg.data.dataset}",
            name=cfg.data.projection,
            split={
                "test": ds.Split.TEST,
                "train": ds.Split.TRAIN,
                "all": ds.Split.ALL,
            }[args.split],
            num_proc=args.num_workers,
            trust_remote_code=True,
        ).with_format("torch"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    config_FRD = dict(feature="penultimate", agg_type="subsample")
    config_FRID = dict(feature="penultimate", agg_type="depth")
    config_FPD = dict(feature=None, agg_type=None)
    config_FSVD = dict(feature="penultimate", agg_type="depth")
    config_FPVD = dict(feature="penultimate", agg_type="depth")

    cache_file_path = f"cache_real_set_{cfg.data.dataset}_{cfg.data.projection}_{H}x{W}_{args.split}.pkl"
    if Path(cache_file_path).exists():
        print(f'Found cached "{cache_file_path}"')
        cache_real = pickle.load(open(cache_file_path, "rb"))
    else:
        cache_real = defaultdict(list)
        for batch in tqdm(loader_real, desc="Real samples"):
            depth = resize(batch["depth"], (H, W)).to(device)
            xyz = resize(batch["xyz"], (H, W)).to(device)
            rflct = resize(batch["reflectance"], (H, W)).to(device)
            mask = resize(batch["mask"], (H, W)).to(device)
            mask = mask * torch.logical_and(depth > MIN_DEPTH, depth < MAX_DEPTH)

            img = torch.cat([depth, xyz, rflct], dim=1) * mask
            pcd = einops.rearrange(img[:, 1:4], "B C H W -> B (H W) C")

            input_FRD = img  # range & reflectance
            feats_FRD = proj(input_FRD, metrics="FRD", **config_FRD)
            cache_real["feats_FRD"].append(feats_FRD.cpu())

            input_FRID = img[:, :4]  # range only
            feats_FRID = proj(input_FRID, metrics="FRID", **config_FRID)
            cache_real["feats_FRID"].append(feats_FRID.cpu())

            input_FPD = pcd.transpose(1, 2) / KITTI_MAX_DEPTH
            feats_FPD = proj(input_FPD, metrics="FPD", **config_FPD)
            cache_real["feats_FPD"].append(feats_FPD.cpu())

            input_FSVD = pcd
            feats_FSVD = proj(input_FSVD, metrics="FSVD", **config_FSVD)
            cache_real["feats_FSVD"].append(feats_FSVD.cpu())

            input_FPVD = pcd
            feats_FPVD = proj(input_FPVD, metrics="FPVD", **config_FPVD)
            cache_real["feats_FPVD"].append(feats_FPVD.cpu())

            for point_cloud in pcd:
                hist = r2flow.metrics.bev.point_cloud_to_histogram(point_cloud)[None]
                cache_real["hists_BEV"].append(hist.cpu())

        for key, value in cache_real.items():
            cache_real[key] = torch.cat(value, dim=0).numpy()

        pickle.dump(cache_real, open(cache_file_path, "wb"))

    for k, v in cache_real.items():
        print(f"{k}:\t{v.shape}")

    # =====================================================
    # gen set
    # =====================================================

    print(f'Sample directory: "{args.sample_dir}"')

    loader_gen = DataLoader(
        dataset=r2flow.utils.training.IndexedSamples(args.sample_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    cache_gen = defaultdict(list)

    def has_rflct(img):
        return img.shape[1] == 5

    for _, batch in tqdm(loader_gen, desc="Generated samples"):
        depth = resize(batch["depth"], (H, W)).to(device)
        xyz = resize(batch["xyz"], (H, W)).to(device)
        rflct = resize(batch["reflectance"], (H, W)).to(device)
        mask = torch.logical_and(depth > MIN_DEPTH, depth < MAX_DEPTH)

        img = torch.cat([depth, xyz, rflct], dim=1) * mask
        pcd = einops.rearrange(img[:, 1:4], "B C H W -> B (H W) C")

        if has_rflct(img):
            input_FRD = img  # range & reflectance
            feats_FRD = proj(input_FRD, metrics="FRD", **config_FRD)
            cache_gen["feats_FRD"].append(feats_FRD.cpu())

        input_FRID = img[:, :4]  # range only
        feats_FRID = proj(input_FRID, metrics="FRID", **config_FRID)
        cache_gen["feats_FRID"].append(feats_FRID.cpu())

        input_FPD = pcd.transpose(1, 2) / KITTI_MAX_DEPTH
        feats_FPD = proj(input_FPD, metrics="FPD", **config_FPD)
        cache_gen["feats_FPD"].append(feats_FPD.cpu())

        input_FSVD = pcd
        feats_FSVD = proj(input_FSVD, metrics="FSVD", **config_FSVD)
        cache_gen["feats_FSVD"].append(feats_FSVD.cpu())

        input_FPVD = pcd
        feats_FPVD = proj(input_FPVD, metrics="FPVD", **config_FPVD)
        cache_gen["feats_FPVD"].append(feats_FPVD.cpu())

        for point_cloud in pcd:
            hist = r2flow.metrics.bev.point_cloud_to_histogram(point_cloud)[None]
            cache_gen["hists_BEV"].append(hist.cpu())

    for key, value in cache_gen.items():
        cache_gen[key] = torch.cat(value, dim=0).numpy()

    for k, v in cache_gen.items():
        print(f"{k}:\t{v.shape}")

    # =====================================================
    # evaluation
    # =====================================================

    torch.cuda.empty_cache()

    results = dict(
        scores=dict(),
        info=dict(
            num_real=len(cache_real["feats_FPD"]),
            num_gen=len(cache_gen["feats_FPD"]),
            split=args.split,
            directory=str(Path(args.sample_dir).resolve()),
        ),
    )

    for metrics in tqdm(proj.available_metrics(), desc="Evaluation"):
        key = f"feats_{metrics}"
        if key in cache_gen:
            results["scores"][metrics] = {}
            results["scores"][metrics]["Frechet Distance"] = (
                r2flow.metrics.distribution.compute_frechet_distance(
                    cache_real[key], cache_gen[key]
                )
            )
            results["scores"][metrics]["Squared MMD"] = (
                r2flow.metrics.distribution.compute_squared_mmd(
                    cache_real[key], cache_gen[key]
                )
            )

    perm = list(range(len(cache_real["hists_BEV"])))
    random.Random(0).shuffle(perm)
    perm = perm[:10_000]

    results["scores"]["BEV-JSD"] = r2flow.metrics.bev.compute_jsd_2d(
        torch.from_numpy(cache_real["hists_BEV"][perm]).to(device).float(),
        torch.from_numpy(cache_gen["hists_BEV"]).to(device).float(),
    )

    results["scores"]["BEV-MMD"] = r2flow.metrics.bev.compute_mmd_2d(
        torch.from_numpy(cache_real["hists_BEV"][perm]).to(device).float(),
        torch.from_numpy(cache_gen["hists_BEV"]).to(device).float(),
    )

    print(results)

    if args.export_name is not None:
        save_path = Path(args.export_name + ".json").resolve()
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f'Saved to "{save_path}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--export_name", type=str, default=None)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    evaluate(args)
