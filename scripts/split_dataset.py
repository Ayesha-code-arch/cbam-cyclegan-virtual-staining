"""Split extracted per-domain patches into train/val subsets at a fixed ratio.

Expects a directory layout of `<root>/ISH/*.jpg` and `<root>/HE/*.jpg`
(non-overlapping tissue patches already tiled from whole-slide images)
and writes `<root>/train/<domain>` and `<root>/val/<domain>`. Test-set
patches should be tiled from held-out donor slides directly, so that the
split stays at the donor level as described in the paper.

Usage:
    python scripts/split_dataset.py --root /path/to/patches_512 --val_split 0.15
"""

import argparse
import os
import random
import shutil


def split_domain(src_dir: str, train_dir: str, val_dir: str, val_split: float, seed: int) -> None:
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    rng = random.Random(seed)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(".jpg")]
    rng.shuffle(files)

    val_count = int(len(files) * val_split)
    val_files = set(files[:val_count])

    for filename in files:
        destination = val_dir if filename in val_files else train_dir
        shutil.copy2(os.path.join(src_dir, filename), os.path.join(destination, filename))

    print(f"{os.path.basename(src_dir)}: {len(files) - val_count} train / {val_count} val")


def main():
    parser = argparse.ArgumentParser(description="Split ISH/HE patches into train/val")
    parser.add_argument("--root", required=True, help="Directory containing ISH/ and HE/ subfolders")
    parser.add_argument("--domains", nargs="+", default=["ISH", "HE"])
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for domain in args.domains:
        source = os.path.join(args.root, domain)
        if not os.path.isdir(source):
            print(f"skip: {source} not found")
            continue
        split_domain(
            source,
            os.path.join(args.root, "train", domain),
            os.path.join(args.root, "val", domain),
            args.val_split,
            args.seed,
        )


if __name__ == "__main__":
    main()
