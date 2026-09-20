"""
split_pkface.py
================
Split the PK-face dataset into train/test sets.

  - Removes singletons (identities with only 1 image).
  - Train: 1 random image per identity.
  - Test:  remaining images per identity.

Usage:
    python split_pkface.py --input /path/to/PK-face --output /path/to/PK-face_split
    python split_pkface.py --input D:\\PK-face --output D:\\PK-face_split --seed 123
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


def collect_identities(data_dir: Path) -> dict[str, list[Path]]:
    """Group images by identity. Identity = everything before the last _N.jpg."""
    identities: dict[str, list[Path]] = defaultdict(list)
    for img in sorted(data_dir.glob("*.jpg")):
        stem = img.stem  # e.g. pa16_1949_0
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            identity_key = parts[0]  # e.g. pa16_1949
        else:
            identity_key = stem
        identities[identity_key].append(img)
    return dict(identities)


def split_dataset(input_dir: Path, output_dir: Path, seed: int = 42) -> None:
    rng = random.Random(seed)

    identities = collect_identities(input_dir)
    print(f"Found {len(identities)} total identities in {input_dir}")

    # Filter out singletons
    multi = {k: v for k, v in identities.items() if len(v) >= 2}
    singletons = {k: v for k, v in identities.items() if len(v) == 1}
    print(f"Singletons (removed): {len(singletons)}")
    print(f"Multi-image identities (kept): {len(multi)}")

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    total_train = 0
    total_test = 0

    for identity_key in sorted(multi):
        images = multi[identity_key]
        rng.shuffle(images)

        train_img = images[0]
        test_imgs = images[1:]

        # Copy train image
        shutil.copy2(train_img, train_dir / train_img.name)
        total_train += 1

        # Copy test images
        for img in test_imgs:
            shutil.copy2(img, test_dir / img.name)
            total_test += 1

        prefix = identity_key.rsplit("_", 1)[0] if "_" in identity_key else identity_key
        manifest_rows.append({
            "identity": identity_key,
            "source_prefix": prefix,
            "total_images": len(images),
            "train_image": train_img.name,
            "test_images": "|".join(img.name for img in test_imgs),
        })

    # Write manifest
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["identity", "source_prefix", "total_images", "train_image", "test_images"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone!")
    print(f"  Train: {total_train} images ({len(multi)} identities)")
    print(f"  Test:  {total_test} images")
    print(f"  Manifest: {manifest_path}")
    print(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split PK-face into train/test (remove singletons).")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to PK-face dataset directory.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output directory (will create train/ and test/ subdirs).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory.")
        return

    split_dataset(input_dir, output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
