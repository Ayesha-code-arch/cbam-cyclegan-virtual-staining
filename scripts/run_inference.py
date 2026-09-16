"""Translate a folder of JPG patches with a trained CBAM-CycleGAN checkpoint.

Edit the paths and settings below, then run:
    python scripts/run_inference.py

Output images are written to OUTPUT_ROOT/EXPERIMENT_NAME, one JPG per
input file, keeping the original filename.
"""

import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.networks import Generator, strip_compile_prefix

# ---- edit these ------------------------------------------------------------
CHECKPOINT_PATH = r"C:\path\to\ckpt_0020.pth"
INPUT_DIR = r"C:\path\to\input_patches"
DIRECTION = "ish2he"            # "ish2he" (G_AB) or "he2ish" (G_BA)
EXPERIMENT_NAME = "epoch20_ish2he"
OUTPUT_ROOT = "experiment_outputs"
IMAGE_SIZE = 512
JPEG_QUALITY = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------------------------------------


def load_generator(checkpoint_path: str, direction: str, device: str) -> Generator:
    generator = Generator(ngf=64, n_blocks=9, cbam_every=3).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    key = "G_AB" if direction == "ish2he" else "G_BA"
    generator.load_state_dict(strip_compile_prefix(state[key]))
    generator.eval()
    return generator


@torch.no_grad()
def translate_folder(generator: Generator, input_dir: str, output_dir: str,
                      image_size: int, device: str, quality: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    transform = T.Compose([
        T.Resize((image_size, image_size), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    files = sorted(Path(input_dir).glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No .jpg files found in {input_dir}")

    for path in tqdm(files, desc="translating"):
        image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        output = (generator(image).squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
        T.ToPILImage()(output).save(os.path.join(output_dir, path.name), format="JPEG", quality=quality)

    print(f"saved {len(files)} images to {output_dir}")


def main():
    output_dir = os.path.join(OUTPUT_ROOT, EXPERIMENT_NAME)
    generator = load_generator(CHECKPOINT_PATH, DIRECTION, DEVICE)
    translate_folder(generator, INPUT_DIR, output_dir, IMAGE_SIZE, DEVICE, JPEG_QUALITY)


if __name__ == "__main__":
    main()
