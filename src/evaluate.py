"""Held-out evaluation for one or more checkpoints.

Computes FID and KID in both translation directions, LPIPS for
ISH -> H&E, and cycle SSIM / PSNR for both domains, on the full
validation or test cohort — matching the protocol used to select
checkpoints in the paper (Sect. 2.5).

Usage:
    python -m src.evaluate --config configs/default.yaml \
        --checkpoints outputs/checkpoints/ckpt_0020_best_val.pth \
                       outputs/checkpoints/ckpt_0042_latest.pth \
        --split test --out results/test_eval.csv
"""

import argparse
import csv
import json
import os
from pathlib import Path

import torch

from .config import load_config
from .metrics import (
    compute_fid,
    compute_kid,
    compute_lpips,
    cycle_ssim_psnr,
    load_inception,
    run_inference,
)
from .networks import Generator, strip_compile_prefix

FIELDS = [
    "checkpoint", "split", "epoch",
    "fid_ISH2HE", "fid_HE2ISH", "kid_ISH2HE", "kid_HE2ISH",
    "lpips_ISH2HE", "cycle_ssim_ISH", "cycle_psnr_ISH",
    "cycle_ssim_HE", "cycle_psnr_HE",
]


def evaluate_checkpoint(checkpoint_path: str, cfg, split: str, device: torch.device) -> dict:
    root = cfg["data.data_root"]
    size = cfg["data.image_size"]
    split_ish = os.path.join(root, split, "ISH")
    split_he = os.path.join(root, split, "HE")

    tag = Path(checkpoint_path).stem + f"_{split}"
    fake_he = os.path.join(cfg["output.results_dir"], tag, "fake_HE")
    fake_ish = os.path.join(cfg["output.results_dir"], tag, "fake_ISH")

    g_ab = Generator(cfg["model.input_nc"], cfg["model.output_nc"],
                      cfg["model.ngf"], cfg["model.n_blocks"], cfg["model.cbam_every"]).to(device)
    g_ba = Generator(cfg["model.input_nc"], cfg["model.output_nc"],
                      cfg["model.ngf"], cfg["model.n_blocks"], cfg["model.cbam_every"]).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    g_ab.load_state_dict(strip_compile_prefix(state["G_AB"]))
    g_ba.load_state_dict(strip_compile_prefix(state["G_BA"]))
    epoch = state.get("epoch", "?")

    run_inference(g_ab, split_ish, fake_he, size, device)
    run_inference(g_ba, split_he, fake_ish, size, device)

    inception = load_inception(device)
    fid_ab = compute_fid(split_he, fake_he, inception, device)
    fid_ba = compute_fid(split_ish, fake_ish, inception, device)
    kid_ab, _ = compute_kid(split_he, fake_he)
    kid_ba, _ = compute_kid(split_ish, fake_ish)

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        lpips_ab = compute_lpips(g_ab, split_ish, split_he, lpips_fn, size, device)
    except ImportError:
        lpips_ab = -1.0

    ssim_ish, psnr_ish = cycle_ssim_psnr(g_ab, g_ba, split_ish, size, device)
    ssim_he, psnr_he = cycle_ssim_psnr(g_ba, g_ab, split_he, size, device)

    return {
        "checkpoint": Path(checkpoint_path).name,
        "split": split,
        "epoch": epoch,
        "fid_ISH2HE": fid_ab, "fid_HE2ISH": fid_ba,
        "kid_ISH2HE": kid_ab, "kid_HE2ISH": kid_ba,
        "lpips_ISH2HE": lpips_ab,
        "cycle_ssim_ISH": ssim_ish, "cycle_psnr_ISH": psnr_ish,
        "cycle_ssim_HE": ssim_he, "cycle_psnr_HE": psnr_he,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CBAM-CycleGAN checkpoints")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--out", default="outputs/results/eval_results.csv")
    parser.add_argument("--set", nargs="*", default=[], metavar="key=value")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results = []
    for checkpoint_path in args.checkpoints:
        print(f"evaluating {checkpoint_path} on {args.split}")
        result = evaluate_checkpoint(checkpoint_path, cfg, args.split, device)
        results.append(result)
        print(json.dumps(result, indent=2, default=str))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
