"""Unpaired patch dataset and generator replay buffer."""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class PatchDataset(Dataset):
    """Unpaired ISH / H&E patch loader.

    Domain A (ISH) is indexed sequentially per epoch; domain B (H&E) is
    sampled at random, following the standard unpaired CycleGAN protocol
    since the two domains are not spatially registered.
    """

    def __init__(self, root_a: str, root_b: str, image_size: int = 512,
                 augment: bool = True, max_steps: int | None = None):
        self.files_a = sorted(Path(root_a).glob("*.jpg"))
        self.files_b = sorted(Path(root_b).glob("*.jpg"))
        self.max_steps = max_steps

        base = [
            T.Resize((image_size, image_size), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        if augment:
            self.transform = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), *base])
        else:
            self.transform = T.Compose(base)

    def __len__(self) -> int:
        n = max(len(self.files_a), len(self.files_b))
        return min(n, self.max_steps) if self.max_steps else n

    def _load(self, path: Path) -> torch.Tensor:
        for _ in range(5):
            try:
                return self.transform(Image.open(path).convert("RGB"))
            except (OSError, ValueError):
                path = random.choice(self.files_a)
        raise RuntimeError(f"Unable to load a valid image after retries near {path}")

    def __getitem__(self, idx: int) -> dict:
        return {
            "A": self._load(self.files_a[idx % len(self.files_a)]),
            "B": self._load(self.files_b[random.randint(0, len(self.files_b) - 1)]),
        }


class ImagePool:
    """Replay buffer of previously generated images (Shrivastava et al., 2017).

    Stabilizes discriminator training by occasionally showing it fakes
    from earlier in training rather than only the current batch.
    """

    def __init__(self, size: int = 50):
        self.size = size
        self.pool: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return images

        result = []
        for image in images:
            image = image.unsqueeze(0)
            if len(self.pool) < self.size:
                self.pool.append(image)
                result.append(image)
            elif random.random() > 0.5:
                idx = random.randint(0, self.size - 1)
                result.append(self.pool[idx].clone())
                self.pool[idx] = image
            else:
                result.append(image)
        return torch.cat(result, dim=0)
