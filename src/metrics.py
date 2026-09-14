"""Distributional and self-consistency metrics used for checkpoint selection.

FID/KID are computed on the full held-out cohort rather than a small
subsample, which avoids the known instability of FID at low sample
counts. Cycle SSIM/PSNR are self-consistency measures on same-domain
reconstructions and substitute for paired ground truth, which is
unavailable given the ~20um offset between adjacent ISH/H&E sections.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


@torch.no_grad()
def run_inference(generator, input_dir: str, output_dir: str,
                   image_size: int = 512, device: str = "cuda",
                   batch_size: int = 8) -> int:
    """Translate every patch in `input_dir` and save results to `output_dir`."""
    generator.eval()
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    transform = T.Compose([
        T.Resize((image_size, image_size), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    files = sorted(Path(input_dir).glob("*.jpg"))

    def flush(batch_images, batch_names):
        if not batch_images:
            return
        batch = torch.stack(batch_images).to(device)
        output = (generator(batch).cpu() * 0.5 + 0.5).clamp(0, 1)
        for i, name in enumerate(batch_names):
            T.ToPILImage()(output[i]).save(os.path.join(output_dir, name), format="JPEG", quality=95)

    images, names = [], []
    for path in tqdm(files, desc="Inference", leave=False):
        try:
            images.append(transform(Image.open(path).convert("RGB")))
            names.append(path.name)
            if len(images) >= batch_size:
                flush(images, names)
                images, names = [], []
        except (OSError, ValueError) as exc:
            print(f"  [skip] {path.name}: {exc}")
    flush(images, names)

    generator.train()
    return len(files)


def load_inception(device: str = "cuda"):
    from pytorch_fid.inception import InceptionV3
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    return model


def compute_fid(real_dir: str, fake_dir: str, inception, device: str = "cuda") -> float:
    from pytorch_fid.fid_score import calculate_frechet_distance, get_activations

    def stats(path):
        files = [str(f) for f in Path(path).glob("*.jpg")]
        if not files:
            return None, None
        activations = get_activations(files, inception, 32, 2048, device, num_workers=2)
        return np.mean(activations, axis=0), np.cov(activations, rowvar=False)

    mu1, sigma1 = stats(real_dir)
    mu2, sigma2 = stats(fake_dir)
    if mu1 is None or mu2 is None:
        return float("inf")
    return float(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))


def compute_kid(real_dir: str, fake_dir: str) -> tuple[float, float]:
    import torch_fidelity
    metrics = torch_fidelity.calculate_metrics(
        input1=real_dir, input2=fake_dir,
        kid=True, fid=False, isc=False,
        cuda=torch.cuda.is_available(), verbose=False,
    )
    return (
        float(metrics["kernel_inception_distance_mean"]),
        float(metrics["kernel_inception_distance_std"]),
    )


def compute_lpips(generator, source_dir: str, reference_dir: str, lpips_fn,
                   image_size: int = 512, device: str = "cuda",
                   n_batches: int = 125, batch_size: int = 8) -> float:
    import random
    from torch.utils.data import DataLoader, Dataset

    class PairedForLPIPS(Dataset):
        def __init__(self, dir_a, dir_b):
            self.a = sorted(Path(dir_a).glob("*.jpg"))
            self.b = sorted(Path(dir_b).glob("*.jpg"))
            self.transform = T.Compose([
                T.Resize((image_size, image_size), T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        def __len__(self):
            return max(len(self.a), len(self.b))

        def __getitem__(self, idx):
            img_a = self.transform(Image.open(self.a[idx % len(self.a)]).convert("RGB"))
            img_b = self.transform(Image.open(self.b[random.randint(0, len(self.b) - 1)]).convert("RGB"))
            return img_a, img_b

    generator.eval()
    loader = DataLoader(PairedForLPIPS(source_dir, reference_dir), batch_size=batch_size,
                         shuffle=False, num_workers=2)
    scores = []
    with torch.no_grad():
        for i, (real_source, real_reference) in enumerate(loader):
            if i >= n_batches:
                break
            fake_reference = generator(real_source.to(device))
            scores.append(float(lpips_fn(real_reference.to(device), fake_reference).mean()))
    generator.train()
    return float(np.mean(scores)) if scores else -1.0


def cycle_ssim_psnr(forward_generator, backward_generator, source_dir: str,
                     image_size: int = 512, device: str = "cuda",
                     n_batches: int = 125, batch_size: int = 8) -> tuple[float, float]:
    from torch.utils.data import DataLoader, Dataset
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
    from skimage.metrics import structural_similarity as ssim_metric

    class SingleDomain(Dataset):
        def __init__(self, root, limit):
            self.files = sorted(Path(root).glob("*.jpg"))[:limit]
            self.transform = T.Compose([
                T.Resize((image_size, image_size), T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            return self.transform(Image.open(self.files[idx]).convert("RGB"))

    forward_generator.eval()
    backward_generator.eval()
    loader = DataLoader(SingleDomain(source_dir, n_batches * batch_size), batch_size=batch_size,
                         shuffle=False, num_workers=2)

    ssim_values, psnr_values = [], []
    with torch.no_grad():
        for i, real in enumerate(loader):
            if i >= n_batches:
                break
            real = real.to(device)
            reconstructed = backward_generator(forward_generator(real))
            real_np = (real.cpu().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5).clip(0, 1)
            recon_np = (reconstructed.cpu().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5).clip(0, 1)
            for r, f in zip(real_np, recon_np):
                ssim_values.append(ssim_metric(r, f, data_range=1.0, channel_axis=2))
                psnr_values.append(psnr_metric(r, f, data_range=1.0))

    forward_generator.train()
    backward_generator.train()
    return (
        float(np.mean(ssim_values)) if ssim_values else -1.0,
        float(np.mean(psnr_values)) if psnr_values else -1.0,
    )
