"""
dedup_with_casia.py
=====================
Re-runs the same validated method from the Phase 2 preprocessing
(exact-pixel MD5 hashing via hashlib.md5(img.tobytes()) after OpenCV
decode -- perceptual hashing was tried first and rejected for false-
positiving on pre-aligned images) but now scans BOTH:

    1. the existing clean_dataset/ folders (LFW, CPLFW, CFP-FP, CALFW,
       AgeDB-30) -- already verified zero cross-dataset duplicates among
       themselves, so re-checking them costs nothing and keeps this script
       self-contained/idempotent.
    2. any CASIA-* folder(s) produced by casia_bin_extractor.py.

This is a REAL cross-dataset check this time, unlike the original Phase 2
run. CASIA's eval/*.bin files are benchmark-standard packagings of LFW /
AgeDB-30 / CALFW / CPLFW / CFP-FP-style data, so it's plausible (not
guaranteed) that some CASIA-<name> images are pixel-identical to images
already sitting in the corresponding plain <name> folder.

Non-destructive by design (matches the Phase 2 philosophy of never
modifying original source data): this script does NOT delete anything in
place. It writes a manifest of which files are duplicates of which, and
can optionally export a deduplicated copy to a new folder, preserving each
dataset's original relative structure exactly like the Phase 2 export did.

Priority when a duplicate group spans multiple folders (i.e. which copy
is kept as the "representative" in the deduplicated export):
    AgeDB-30 > LFW > CALFW > CPLFW > CFP-FP > CASIA-* (any)
The five original folders keep their established Phase 2 priority order;
CASIA-* folders are lowest priority across the board since they're a
backup/supplementary source, not the primary tested dataset.

Usage
-----
    # Manifest only (no files copied) -- default, safe/read-only w.r.t.
    # the dataset itself, just writes the CSV:
    python dedup_with_casia.py --dataset-root "C:\\Users\\hasee\\Downloads\\clean_dataset"

    # Also export a deduplicated copy to a new folder:
    python dedup_with_casia.py --dataset-root "C:\\Users\\hasee\\Downloads\\clean_dataset" \\
        --export-to "C:\\Users\\hasee\\Downloads\\clean_dataset_with_casia_deduped"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
from tqdm import tqdm

from casia_bin_extractor import BIN_TO_DATASET_SOURCE as CASIA_DATASET_SOURCES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dedup_with_casia")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

FIXED_DATASET_FOLDERS = ["LFW", "CPLFW", "CFP-FP", "CALFW", "AgeDB-30"]
CASIA_FOLDER_NAMES = list(CASIA_DATASET_SOURCES.values())

# Lower index = higher priority = kept as the representative when a
# duplicate group spans folders. Any folder not listed here (shouldn't
# happen given discover_dataset_folders() below) sorts last.
#
# NOTE: this is a DELIBERATELY different order from FIXED_DATASET_FOLDERS
# above (which is just scan order and doesn't matter for that purpose) --
# do not "simplify" this back to FIXED_DATASET_FOLDERS + CASIA_FOLDER_NAMES,
# that was the original bug (AgeDB-30 ended up lowest priority of the 5
# instead of highest).
DEDUP_PRIORITY_FOLDERS = ["AgeDB-30", "LFW", "CALFW", "CPLFW", "CFP-FP"]
PRIORITY_ORDER = DEDUP_PRIORITY_FOLDERS + CASIA_FOLDER_NAMES


def _priority(dataset_folder: str) -> int:
    try:
        return PRIORITY_ORDER.index(dataset_folder)
    except ValueError:
        return len(PRIORITY_ORDER)


def discover_dataset_folders(dataset_root: Path) -> List[str]:
    """Fixed 5 folders (if present) + any CASIA-* folders actually on disk."""
    present = []
    for name in FIXED_DATASET_FOLDERS + CASIA_FOLDER_NAMES:
        if (dataset_root / name).is_dir():
            present.append(name)
        else:
            logger.debug(f"{name} not present under {dataset_root}, skipping.")
    return present


def hash_image(path: Path) -> str | None:
    img = cv2.imread(str(path))
    if img is None:
        logger.warning(f"Could not decode (corrupt/unsupported?), skipping: {path}")
        return None
    return hashlib.md5(img.tobytes()).hexdigest()


def find_duplicate_groups(dataset_root: Path, folders: List[str]) -> Dict[str, List[Path]]:
    """md5 -> list of file paths sharing that exact pixel content, across ALL given folders."""
    hash_to_paths: Dict[str, List[Path]] = defaultdict(list)

    all_paths = []
    for folder_name in folders:
        folder = dataset_root / folder_name
        all_paths.extend(
            p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

    logger.info(f"Hashing {len(all_paths)} images across {len(folders)} folder(s) ...")
    for path in tqdm(all_paths, desc="MD5 hashing"):
        h = hash_image(path)
        if h is not None:
            hash_to_paths[h].append(path)

    return {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}


def folder_of(path: Path, dataset_root: Path) -> str:
    return path.relative_to(dataset_root).parts[0]


def choose_representative(paths: List[Path], dataset_root: Path) -> Path:
    return min(paths, key=lambda p: _priority(folder_of(p, dataset_root)))


def run_dedup(dataset_root: str, export_to: str | None, manifest_path: str) -> None:
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    folders = discover_dataset_folders(root)
    if not folders:
        raise FileNotFoundError(f"No known dataset folders found under {root}.")
    logger.info(f"Scanning folders: {folders}")

    dup_groups = find_duplicate_groups(root, folders)
    logger.info(f"Found {len(dup_groups)} exact-duplicate group(s).")

    cross_dataset_groups = 0
    rows = []
    keep_paths = set()
    remove_paths = set()

    for md5, paths in dup_groups.items():
        rep = choose_representative(paths, root)
        keep_paths.add(rep)
        group_folders = {folder_of(p, root) for p in paths}
        if len(group_folders) > 1:
            cross_dataset_groups += 1

        for p in paths:
            is_kept = p == rep
            if not is_kept:
                remove_paths.add(p)
            rows.append({
                "md5": md5,
                "path": str(p),
                "dataset_folder": folder_of(p, root),
                "kept": is_kept,
                "group_spans_multiple_datasets": len(group_folders) > 1,
            })

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["md5", "path", "dataset_folder", "kept", "group_spans_multiple_datasets"]
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        f"{cross_dataset_groups} duplicate group(s) span more than one dataset folder "
        f"(this is the check the original Phase 2 dedup couldn't do, since CASIA wasn't "
        f"part of that run). {len(remove_paths)} total files would be dropped as duplicates. "
        f"Manifest written to {manifest_path}."
    )

    if export_to is None:
        logger.info("--export-to not given, so no files were copied (manifest-only run).")
        return

    export_root = Path(export_to)
    n_copied, n_skipped_as_dup = 0, 0
    all_paths = []
    for folder_name in folders:
        folder = root / folder_name
        all_paths.extend(
            p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

    for path in tqdm(all_paths, desc="Exporting deduplicated copy"):
        if path in remove_paths:
            n_skipped_as_dup += 1
            continue
        rel = path.relative_to(root)
        dest = export_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        n_copied += 1

    logger.info(
        f"Export complete: {n_copied} images copied to {export_root}, "
        f"{n_skipped_as_dup} exact-duplicate images skipped."
    )


def main():
    parser = argparse.ArgumentParser(
        description="MD5 exact-pixel dedup across the existing 5 datasets plus any CASIA-* extracts."
    )
    parser.add_argument(
        "--dataset-root", type=str, default=r"C:\Users\hasee\Downloads\clean_dataset",
        help="Root folder containing LFW/CPLFW/CFP-FP/CALFW/AgeDB-30 and any CASIA-* subfolders.",
    )
    parser.add_argument(
        "--export-to", type=str, default=None,
        help="If given, copies a deduplicated set (one representative per exact-duplicate "
             "group, original relative folder structure preserved) here. If omitted, this "
             "run only writes the manifest CSV and touches nothing on disk.",
    )
    parser.add_argument(
        "--manifest-path", type=str, default="dedup_with_casia_manifest.csv",
        help="Where to write the kept/removed manifest CSV.",
    )
    args = parser.parse_args()
    run_dedup(args.dataset_root, args.export_to, args.manifest_path)


if __name__ == "__main__":
    main()
