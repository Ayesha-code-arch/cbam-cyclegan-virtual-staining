"""Training entry point for CBAM-CycleGAN.

Every epoch runs a full training pass plus a cheap generator-loss
validation (500 patches) used for early stopping and checkpoint
selection. Full distributional evaluation (FID / KID / LPIPS) is run
every `eval.full_eval_freq` epochs, since it is far more expensive than
the proxy loss and, as shown in the paper, the two do not always agree.

Usage:
    python -m src.train --config configs/default.yaml \
        --set data.data_root=/path/to/patches_512
"""

import csv
import glob
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from .config import build_arg_parser, load_config
from .datasets import ImagePool, PatchDataset
from .losses import cycle_edge_loss, ssim_loss
from .metrics import compute_fid, compute_kid, compute_lpips, run_inference
from .networks import Discriminator, Generator, strip_compile_prefix

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.G_AB = Generator(cfg["model.input_nc"], cfg["model.output_nc"],
                               cfg["model.ngf"], cfg["model.n_blocks"],
                               cfg["model.cbam_every"]).to(device)
        self.G_BA = Generator(cfg["model.input_nc"], cfg["model.output_nc"],
                               cfg["model.ngf"], cfg["model.n_blocks"],
                               cfg["model.cbam_every"]).to(device)
        self.D_A = Discriminator(cfg["model.input_nc"], cfg["model.ndf"]).to(device)
        self.D_B = Discriminator(cfg["model.input_nc"], cfg["model.ndf"]).to(device)

        self.pool_a = ImagePool(cfg["optim.pool_size"])
        self.pool_b = ImagePool(cfg["optim.pool_size"])

        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        except ImportError:
            self.lpips_fn = None

        self.gan_loss = nn.MSELoss()
        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()

        self.opt_g = optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=cfg["optim.lr_g"], betas=(cfg["optim.beta1"], 0.999),
        )
        self.opt_d = optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=cfg["optim.lr_d"], betas=(cfg["optim.beta1"], 0.999),
        )

        n_flat, n_decay = cfg["optim.n_epochs"], cfg["optim.n_epochs_decay"]

        def lr_lambda(epoch):
            return 1.0 if epoch < n_flat else max(0.0, 1.0 - (epoch - n_flat) / float(n_decay))

        self.sched_g = optim.lr_scheduler.LambdaLR(self.opt_g, lr_lambda)
        self.sched_d = optim.lr_scheduler.LambdaLR(self.opt_d, lr_lambda)
        self.scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

        self.start_epoch = 0
        self.total_iters = 0
        self.best_val_loss = float("inf")
        self.best_fid = float("inf")
        self.patience_count = 0

        for directory in (cfg["output.checkpoint_dir"], cfg["output.sample_dir"], cfg["output.results_dir"]):
            os.makedirs(directory, exist_ok=True)

        log_path = cfg["output.log_path"]
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "epoch", "iter", "loss_G", "loss_D_A", "loss_D_B",
                    "loss_cycle", "loss_ssim", "loss_edge", "loss_idt",
                    "val_loss_G", "val_D_A", "val_D_B",
                    "fid_ISH2HE", "kid_ISH2HE", "lpips_ISH2HE",
                    "lr_G", "lr_D",
                ])

    # ---- checkpointing ---------------------------------------------------

    def save_checkpoint(self, epoch: int, tag: str = "latest") -> None:
        path = os.path.join(self.cfg["output.checkpoint_dir"], f"ckpt_{epoch:04d}_{tag}.pth")
        tmp_path = path + ".tmp"
        torch.save({
            "epoch": epoch,
            "iter": self.total_iters,
            "best_val_loss": self.best_val_loss,
            "best_fid": self.best_fid,
            "patience": self.patience_count,
            "G_AB": self.G_AB.state_dict(),
            "G_BA": self.G_BA.state_dict(),
            "D_A": self.D_A.state_dict(),
            "D_B": self.D_B.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "sched_g": self.sched_g.state_dict(),
            "sched_d": self.sched_d.state_dict(),
            "scaler": self.scaler.state_dict(),
        }, tmp_path)
        os.replace(tmp_path, path)

        with open(os.path.join(self.cfg["output.checkpoint_dir"], "latest.json"), "w") as f:
            json.dump({"path": path, "epoch": epoch}, f)

        # Keep only the two most recent routine checkpoints; best_val / best_fid
        # tags are matched by suffix and never pruned.
        routine = sorted(glob.glob(os.path.join(self.cfg["output.checkpoint_dir"], "ckpt_*_latest.pth")))
        for old_path in routine[:-2]:
            os.remove(old_path)

    def load_checkpoint(self) -> bool:
        latest_json = os.path.join(self.cfg["output.checkpoint_dir"], "latest.json")
        if not os.path.exists(latest_json):
            return False

        with open(latest_json) as f:
            info = json.load(f)
        state = torch.load(info["path"], map_location=self.device)

        self.G_AB.load_state_dict(strip_compile_prefix(state["G_AB"]))
        self.G_BA.load_state_dict(strip_compile_prefix(state["G_BA"]))
        self.D_A.load_state_dict(strip_compile_prefix(state["D_A"]))
        self.D_B.load_state_dict(strip_compile_prefix(state["D_B"]))
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.sched_g.load_state_dict(state["sched_g"])
        self.sched_d.load_state_dict(state["sched_d"])
        self.scaler.load_state_dict(state["scaler"])

        self.start_epoch = state["epoch"] + 1
        self.total_iters = state.get("iter", 0)
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        self.best_fid = state.get("best_fid", float("inf"))
        self.patience_count = state.get("patience", 0)
        return True

    # ---- training ----------------------------------------------------------

    def train_step(self, real_a: torch.Tensor, real_b: torch.Tensor) -> dict:
        cfg = self.cfg
        self.opt_g.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            fake_b = self.G_AB(real_a)
            fake_a = self.G_BA(real_b)
            recon_a = self.G_BA(fake_b)
            recon_b = self.G_AB(fake_a)
            idt_a = self.G_BA(real_a)
            idt_b = self.G_AB(real_b)

            pred_fake_b = self.D_B(fake_b)
            pred_fake_a = self.D_A(fake_a)
            loss_gan = (self.gan_loss(pred_fake_b, torch.ones_like(pred_fake_b)) +
                        self.gan_loss(pred_fake_a, torch.ones_like(pred_fake_a)))

            loss_cycle = (self.cycle_loss(recon_a, real_a) +
                          self.cycle_loss(recon_b, real_b)) * cfg["loss.lambda_cycle"]
            loss_ssim = (ssim_loss(recon_a, real_a) +
                         ssim_loss(recon_b, real_b)) * cfg["loss.lambda_cycle_ssim"]
            loss_edge = (cycle_edge_loss(recon_a, real_a) +
                         cycle_edge_loss(recon_b, real_b)) * cfg["loss.lambda_cycle_edge"]
            loss_idt = (self.identity_loss(idt_a, real_a) +
                        self.identity_loss(idt_b, real_b)) * cfg["loss.lambda_identity"]

            loss_g = loss_gan + loss_cycle + loss_ssim + loss_edge + loss_idt

        self.scaler.scale(loss_g).backward()
        if cfg["optim.grad_clip"]:
            self.scaler.unscale_(self.opt_g)
            nn.utils.clip_grad_norm_(
                list(self.G_AB.parameters()) + list(self.G_BA.parameters()), cfg["optim.grad_clip"])
        self.scaler.step(self.opt_g)

        self.opt_d.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            fake_a_pool = self.pool_a.query(fake_a.detach())
            fake_b_pool = self.pool_b.query(fake_b.detach())

            pred_real_a = self.D_A(real_a)
            pred_fake_a = self.D_A(fake_a_pool)
            pred_real_b = self.D_B(real_b)
            pred_fake_b = self.D_B(fake_b_pool)

            # One-sided label smoothing (Salimans et al., 2016): real targets at 0.9.
            real_label_a = torch.full_like(pred_real_a, 0.9)
            real_label_b = torch.full_like(pred_real_b, 0.9)

            loss_d_a = 0.5 * (self.gan_loss(pred_real_a, real_label_a) +
                              self.gan_loss(pred_fake_a, torch.zeros_like(pred_fake_a)))
            loss_d_b = 0.5 * (self.gan_loss(pred_real_b, real_label_b) +
                              self.gan_loss(pred_fake_b, torch.zeros_like(pred_fake_b)))

        self.scaler.scale(loss_d_a + loss_d_b).backward()
        if cfg["optim.grad_clip"]:
            self.scaler.unscale_(self.opt_d)
            nn.utils.clip_grad_norm_(
                list(self.D_A.parameters()) + list(self.D_B.parameters()), cfg["optim.grad_clip"])
        self.scaler.step(self.opt_d)
        self.scaler.update()

        return {
            "loss_G": float(loss_g.detach()), "loss_D_A": float(loss_d_a.detach()),
            "loss_D_B": float(loss_d_b.detach()), "loss_cyc": float(loss_cycle.detach()),
            "loss_ssim": float(loss_ssim.detach()), "loss_edge": float(loss_edge.detach()),
            "loss_idt": float(loss_idt.detach()),
            "real_a": real_a.detach(), "real_b": real_b.detach(),
            "fake_a": fake_a.detach(), "fake_b": fake_b.detach(),
            "recon_a": recon_a.detach(), "recon_b": recon_b.detach(),
        }

    @torch.no_grad()
    def quick_val(self, val_loader, n_batches: int) -> tuple[float, float, float]:
        """Cheap generator-loss validation used for early stopping and
        best-checkpoint tagging every epoch (no FID/KID here)."""
        for module in (self.G_AB, self.G_BA, self.D_A, self.D_B):
            module.eval()

        g_losses, d_a_losses, d_b_losses = [], [], []
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            real_a = batch["A"].to(self.device)
            real_b = batch["B"].to(self.device)

            fake_b = self.G_AB(real_a)
            fake_a = self.G_BA(real_b)
            recon_a = self.G_BA(fake_b)
            recon_b = self.G_AB(fake_a)
            idt_a = self.G_BA(real_a)
            idt_b = self.G_AB(real_b)

            pred_fake_b = self.D_B(fake_b)
            pred_fake_a = self.D_A(fake_a)
            loss_gan = (self.gan_loss(pred_fake_b, torch.ones_like(pred_fake_b)) +
                        self.gan_loss(pred_fake_a, torch.ones_like(pred_fake_a)))
            loss_cycle = (self.cycle_loss(recon_a, real_a) +
                          self.cycle_loss(recon_b, real_b)) * self.cfg["loss.lambda_cycle"]
            loss_idt = (self.identity_loss(idt_a, real_a) +
                        self.identity_loss(idt_b, real_b)) * self.cfg["loss.lambda_identity"]
            g_losses.append(float(loss_gan + loss_cycle + loss_idt))

            pred_real_a = self.D_A(real_a)
            pred_real_b = self.D_B(real_b)
            d_a_losses.append(float(0.5 * (
                self.gan_loss(pred_real_a, torch.ones_like(pred_real_a)) +
                self.gan_loss(pred_fake_a, torch.zeros_like(pred_fake_a)))))
            d_b_losses.append(float(0.5 * (
                self.gan_loss(pred_real_b, torch.ones_like(pred_real_b)) +
                self.gan_loss(pred_fake_b, torch.zeros_like(pred_fake_b)))))

        for module in (self.G_AB, self.G_BA, self.D_A, self.D_B):
            module.train()

        return float(np.mean(g_losses)), float(np.mean(d_a_losses)), float(np.mean(d_b_losses))

    def run_full_eval(self, val_loader, epoch: int) -> tuple[float, float, float]:
        val_ish = os.path.join(self.cfg["data.data_root"], "val", "ISH")
        val_he = os.path.join(self.cfg["data.data_root"], "val", "HE")
        fake_he = os.path.join(self.cfg["output.results_dir"], "val_fake_HE")
        fake_ish = os.path.join(self.cfg["output.results_dir"], "val_fake_ISH")
        size = self.cfg["data.image_size"]

        run_inference(self.G_AB, val_ish, fake_he, size, self.device)
        run_inference(self.G_BA, val_he, fake_ish, size, self.device)

        inception = load_inception_cached(self.device)
        fid_ab = compute_fid(val_he, fake_he, inception, self.device)
        kid_ab, _ = compute_kid(val_he, fake_he)
        lpips_ab = compute_lpips(self.G_AB, val_ish, val_he, self.lpips_fn, size, self.device,
                                  self.cfg["eval.val_batches"]) if self.lpips_fn else -1.0

        if fid_ab < self.best_fid:
            self.best_fid = fid_ab
            self.save_checkpoint(epoch, "best_fid")

        return fid_ab, kid_ab, lpips_ab

    def save_samples(self, step_output: dict, epoch: int) -> None:
        def denorm(t):
            return (t.cpu() * 0.5 + 0.5).clamp(0, 1)

        grid = make_grid([
            denorm(step_output["real_a"][:1].squeeze(0)),
            denorm(step_output["fake_b"][:1].squeeze(0)),
            denorm(step_output["recon_a"][:1].squeeze(0)),
            denorm(step_output["real_b"][:1].squeeze(0)),
            denorm(step_output["fake_a"][:1].squeeze(0)),
            denorm(step_output["recon_b"][:1].squeeze(0)),
        ], nrow=3)
        save_image(grid, os.path.join(self.cfg["output.sample_dir"], f"epoch_{epoch:04d}.jpg"))

    def train(self, train_loader, val_loader) -> None:
        total_epochs = self.cfg["optim.n_epochs"] + self.cfg["optim.n_epochs_decay"]

        for epoch in range(self.start_epoch, total_epochs):
            for module in (self.G_AB, self.G_BA, self.D_A, self.D_B):
                module.train()

            epoch_losses = defaultdict(list)
            last_output = None
            progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{total_epochs}", total=len(train_loader))

            for batch in progress:
                real_a = batch["A"].to(self.device)
                real_b = batch["B"].to(self.device)
                last_output = self.train_step(real_a, real_b)
                self.total_iters += 1
                for key, value in last_output.items():
                    if key.startswith("loss_"):
                        epoch_losses[key].append(value)
                progress.set_postfix(G=f"{last_output['loss_G']:.3f}",
                                      D=f"{last_output['loss_D_A']:.2f}|{last_output['loss_D_B']:.2f}")

            self.sched_g.step()
            self.sched_d.step()

            if last_output is not None:
                self.save_samples(last_output, epoch + 1)

            val_g, val_d_a, val_d_b = self.quick_val(val_loader, self.cfg["eval.val_batches"])

            if val_g < self.best_val_loss:
                self.best_val_loss = val_g
                self.patience_count = 0
                self.save_checkpoint(epoch + 1, "best_val")
            else:
                self.patience_count += 1

            fid = kid = lpips_score = float("inf")
            if (epoch + 1) % self.cfg["eval.full_eval_freq"] == 0:
                fid, kid, lpips_score = self.run_full_eval(val_loader, epoch + 1)

            self.save_checkpoint(epoch + 1)

            means = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
            with open(self.cfg["output.log_path"], "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1, self.total_iters,
                    means.get("loss_G", ""), means.get("loss_D_A", ""), means.get("loss_D_B", ""),
                    means.get("loss_cyc", ""), means.get("loss_ssim", ""),
                    means.get("loss_edge", ""), means.get("loss_idt", ""),
                    val_g, val_d_a, val_d_b, fid, kid, lpips_score,
                    self.sched_g.get_last_lr()[0], self.sched_d.get_last_lr()[0],
                ])

            print(f"epoch {epoch + 1}: val_G={val_g:.4f}  val_D={val_d_a:.3f}|{val_d_b:.3f}  "
                  f"patience={self.patience_count}/{self.cfg['eval.early_stop_patience']}")

            if self.patience_count >= self.cfg["eval.early_stop_patience"]:
                print(f"early stopping at epoch {epoch + 1}")
                break

        print(f"training complete, best_val_G={self.best_val_loss:.4f}  best_fid={self.best_fid:.2f}")


_inception_cache = {}


def load_inception_cached(device):
    if device not in _inception_cache:
        from .metrics import load_inception
        _inception_cache[device] = load_inception(device)
    return _inception_cache[device]


def main():
    parser = build_arg_parser("Train CBAM-CycleGAN for ISH <-> H&E virtual staining")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    root = cfg["data.data_root"]
    train_a = os.path.join(root, "train", "ISH")
    train_b = os.path.join(root, "train", "HE")
    val_a = os.path.join(root, "val", "ISH")
    val_b = os.path.join(root, "val", "HE")

    for path in (train_a, train_b, val_a, val_b):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Expected patch directory not found: {path}")

    train_ds = PatchDataset(train_a, train_b, cfg["data.image_size"],
                             augment=True, max_steps=cfg["data.steps_per_epoch"])
    val_ds = PatchDataset(val_a, val_b, cfg["data.image_size"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg["data.batch_size"], shuffle=True,
                               num_workers=cfg["data.num_workers"], pin_memory=True,
                               drop_last=True, persistent_workers=cfg["data.num_workers"] > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg["data.batch_size"], shuffle=False,
                             num_workers=cfg["data.num_workers"], pin_memory=True,
                             persistent_workers=cfg["data.num_workers"] > 0)

    trainer = Trainer(cfg, device)
    if trainer.load_checkpoint():
        print(f"resumed from epoch {trainer.start_epoch}")
    else:
        print("starting from scratch")

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
