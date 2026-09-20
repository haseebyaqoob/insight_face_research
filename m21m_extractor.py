"""
m21m_extractor.py
==================
Split a folder of person-subfolders into train/test sets.

For each person subfolder: picks 1 image for train, rest go to test.

Usage:
    python m21m_extractor.py --input /path/to/persons --output /path/to/output
    python m21m_extractor.py --input D:\\faces\\ms1m --output D:\\faces\\ms1m_split --limit 100
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m21m_extractor")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def run(
    input_dir: Path,
    output_dir: Path,
    train_count: int = 1,
    seed: int = 42,
    limit: int = None,
) -> dict:
    person_folders = sorted(
        p for p in input_dir.iterdir() if p.is_dir()
    )
    if limit:
        person_folders = person_folders[:limit]

    if not person_folders:
        raise ValueError(f"No person subfolders found in {input_dir}")

    logger.info(f"Found {len(person_folders)} person folders in {input_dir}")

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    n_train = 0
    n_test = 0
    n_skipped = 0

    for person_folder in person_folders:
        images = collect_images(person_folder)
        if not images:
            n_skipped += 1
            continue

        rng.shuffle(images)
        train_imgs = images[:train_count]
        test_imgs = images[train_count:]

        person_name = person_folder.name

        for img in train_imgs:
            dest = train_dir / f"{person_name}_{img.name}"
            shutil.copy2(img, dest)
            n_train += 1

        for img in test_imgs:
            dest = test_dir / f"{person_name}_{img.name}"
            shutil.copy2(img, dest)
            n_test += 1

    stats = {"train": n_train, "test": n_test, "skipped_empty": n_skipped}
    logger.info(
        f"Done. train={n_train}, test={n_test}, skipped(empty folders)={n_skipped}"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Split person-subfolders into train/test (1 per person for train).")
    parser.add_argument("--input", type=str, required=True,
                        help="Folder containing person subfolders with images.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output folder (will create train/ and test/ inside).")
    parser.add_argument("--train-count", type=int, default=1,
                        help="Number of images per person for train (default: 1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of person folders to process (for testing).")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory.")
        return

    run(
        input_dir=input_dir,
        output_dir=Path(args.output),
        train_count=args.train_count,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
