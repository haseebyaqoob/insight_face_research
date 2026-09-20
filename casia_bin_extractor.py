"""
casia_bin_extractor.py
=======================
Parses the InsightFace-format `.bin` verification files bundled inside the
CASIA-WebFace Kaggle package's `eval/` folder (agedb_30.bin, calfw.bin,
cfp_ff.bin, cfp_fp.bin, cplfw.bin, lfw.bin, sllfw.bin, talfw.bin) and
extracts the embedded images to disk as a flat, pre-aligned image folder
per bin file.

Format background
------------------
Each `.bin` file is a pickled 2-tuple: (bins, issame_list)
    - bins: a list of JPEG-encoded image bytes, length == 2 * len(issame_list)
            (verification pairs are stored back-to-back: bins[2*i] and
            bins[2*i+1] form pair i)
    - issame_list: a list[bool] of length N, True if pair i is same-identity

This is the exact format used by the official InsightFace/ArcFace
evaluation scripts. Older files were pickled under Python 2, so bins may
need `pickle.load(f, encoding="bytes")` to load cleanly under Python 3 --
this module tries both.

Images inside these bins are ALREADY the standard ArcFace-aligned 112x112
crop (same alignment target as ARCFACE_DST in embedder.py) -- that's the
whole point of a verification bin, it has to be pipeline-ready for
whatever recognition model is being benchmarked. Treat them exactly like
this project's existing AgeDB-30/CALFW/CPLFW pre-aligned sources: NO
RetinaFace detection/alignment step when embedding them. See
bootstrap_pipeline.py (auto-discovers CASIA-* folders + routes them
through embedder.embed_image(..., is_aligned=True)) and embedder.py's
`is_aligned` parameter.

Dataset-source naming
----------------------
Extracted folders are named "CASIA-<BenchmarkName>" (e.g. "CASIA-LFW"),
deliberately NOT reusing the plain "LFW" / "AgeDB-30" / etc. names already
used by the existing raw/pre-aligned exports. Even though this is
nominally "the same benchmark", it's a different package/download and
must be verified as exact-duplicate content (see dedup_with_casia.py)
rather than assumed identical to what's already in clean_dataset/.

NOT part of current testing
----------------------------
Per project decision, the CASIA package is a backup dataset, not used in
the active LFW+AgeDB-30+CALFW+CPLFW+CFP-FP pipeline right now. This script
exists so extraction is ready to go the moment it's needed -- run it
manually; it is not called automatically by bootstrap_pipeline.py.

Pair labels (issame_list) are written out to a `<bin_name>_pairs.csv`
manifest alongside each output folder for future verification-style eval,
even though the current app only does nearest-neighbour similarity search
and does not consume pair labels today.

Usage
-----
    python casia_bin_extractor.py --eval-dir "D:\\Code\insightface\\CASIA\faces_webface_112x112" \\
        --output-root "D:\\Code\\embedding_model"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("casia_bin_extractor")

# ---------------------------------------------------------------------------
# PLACEHOLDER -- point this at wherever the CASIA package actually lives once
# it's copied off the USB drive (screenshot showed a top-level
# "casia-webface/" folder with an "eval/" subfolder containing the .bin
# files -- --eval-dir should point AT that eval/ subfolder). Output root
# defaults to the same clean_dataset/ root the rest of the pipeline reads
# from, so bootstrap_pipeline.py picks up CASIA-* folders automatically.
# ---------------------------------------------------------------------------
CASIA_EVAL_DIR_PLACEHOLDER = r"D:\Code\insightface\CASIA\faces_webface_112x112"
CASIA_OUTPUT_ROOT_PLACEHOLDER = r"D:\Code\embedding_model\CASIA"

# bin filename (without extension) -> the dataset_source label used
# everywhere downstream (Milvus metadata, bootstrap folder name, dedup
# priority).
BIN_TO_DATASET_SOURCE = {
    "agedb_30": "CASIA-AgeDB-30",
    "calfw": "CASIA-CALFW",
    "cfp_ff": "CASIA-CFP-FF",
    "cfp_fp": "CASIA-CFP-FP",
    "cplfw": "CASIA-CPLFW",
    "lfw": "CASIA-LFW",
    "sllfw": "CASIA-SLLFW",
    "talfw": "CASIA-TALFW",
}

IMAGE_EXT = ".jpg"


def _load_bin_file(bin_path: Path) -> Tuple[List[bytes], List[bool]]:
    """
    Loads an InsightFace-format verification .bin file.
    Tries encoding='bytes' first (handles files originally pickled under
    Python 2, the common case for these benchmark bins), then falls back
    to a plain load for files already re-pickled under Python 3.
    """
    with open(bin_path, "rb") as f:
        raw = f.read()
    last_err: Exception | None = None
    for kwargs in ({"encoding": "bytes"}, {}):
        try:
            bins, issame_list = pickle.loads(raw, **kwargs)
            return list(bins), list(issame_list)
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"Could not unpickle {bin_path} with either py2-bytes or default encoding.") from last_err


def _decode_image(img_bytes: bytes) -> np.ndarray:
    """JPEG bytes -> BGR np.ndarray, same result cv2.imread would give."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None -- corrupt or non-JPEG bin entry.")
    return img


def extract_bin_file(bin_path: Path, output_dir: Path, dataset_source: str) -> dict:
    """
    Extracts every image in one .bin file to `output_dir/img_%05d.jpg`,
    plus a `<bin_stem>_pairs.csv` manifest of (pair_index, image_a, image_b, issame).
    Returns a stats dict for the run summary.

    IMPORTANT: the on-disk file is the ORIGINAL bin bytes, written verbatim
    (no cv2 decode+re-encode round trip). We still decode in-memory to
    compute the exact-pixel MD5 for intra-bin dedup and to validate the
    entry isn't corrupt, but the bytes written to disk are untouched. This
    matters because dedup_with_casia.py later re-reads these files and
    hashes ITS decode to look for cross-dataset duplicates against
    LFW/AgeDB-30/etc. -- if we'd re-encoded here, that re-encode is lossy
    and would make a genuinely-identical source photo hash differently on
    the two sides, causing real duplicates to be silently missed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bins, issame_list = _load_bin_file(bin_path)

    n_expected = len(issame_list) * 2
    if len(bins) != n_expected:
        logger.warning(
            f"{bin_path.name}: expected {n_expected} images for {len(issame_list)} "
            f"pairs, found {len(bins)} -- proceeding with what's present."
        )

    n_written, n_failed, n_intra_bin_dupes = 0, 0, 0
    seen_md5 = set()
    written_indices: set[int] = set()

    for i, img_bytes in enumerate(tqdm(bins, desc=f"Extracting {bin_path.stem}", leave=False)):
        try:
            img = _decode_image(img_bytes)
        except Exception as e:
            n_failed += 1
            logger.debug(f"{bin_path.name} index {i}: decode failed ({e}), skipping.")
            continue

        # Exact-pixel MD5, same method as the validated Phase 2 dedup -- a
        # verification bin legitimately reuses the same photo across
        # multiple pairs, so intra-bin dupes are expected, not an error.
        md5 = hashlib.md5(img.tobytes()).hexdigest()
        if md5 in seen_md5:
            n_intra_bin_dupes += 1
            continue
        seen_md5.add(md5)

        dest = output_dir / f"img_{i:05d}{IMAGE_EXT}"
        try:
            # Write the ORIGINAL bytes verbatim -- do NOT cv2.imwrite(img),
            # which would re-encode (lossy) and break byte/pixel-identity
            # with the source. See docstring above.
            with open(dest, "wb") as f:
                f.write(img_bytes)
        except OSError as e:
            n_failed += 1
            logger.debug(f"{bin_path.name} index {i}: writing {dest} failed ({e}).")
            continue
        n_written += 1
        written_indices.add(i)

    pairs_path = output_dir.parent / f"{bin_path.stem}_pairs.csv"
    n_pairs_dropped = 0
    with open(pairs_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_index", "image_a", "image_b", "issame"])
        for i, issame in enumerate(issame_list):
            a_idx, b_idx = 2 * i, 2 * i + 1
            # Only reference files that actually made it to disk -- a pair
            # whose image was an intra-bin duplicate, failed to decode, or
            # fell outside a truncated bin has no file at that index, and
            # writing its name here would leave the manifest pointing at
            # nothing (silently breaking any future consumer of this CSV).
            if a_idx not in written_indices or b_idx not in written_indices:
                n_pairs_dropped += 1
                continue
            writer.writerow([i, f"img_{a_idx:05d}{IMAGE_EXT}", f"img_{b_idx:05d}{IMAGE_EXT}", issame])

    logger.info(
        f"{bin_path.name} -> {dataset_source}: wrote {n_written}, "
        f"intra-bin exact dupes skipped {n_intra_bin_dupes}, failed {n_failed}, "
        f"pairs dropped from manifest (missing image) {n_pairs_dropped}. "
        f"Pair manifest: {pairs_path}"
    )
    return {
        "dataset_source": dataset_source,
        "written": n_written,
        "intra_bin_dupes": n_intra_bin_dupes,
        "failed": n_failed,
        "pairs_dropped": n_pairs_dropped,
        "pairs_file": str(pairs_path),
    }


def extract_all(eval_dir: str, output_root: str) -> list[dict]:
    eval_path = Path(eval_dir)
    output_path = Path(output_root)

    if not eval_path.is_dir():
        raise FileNotFoundError(
            f"CASIA eval/ directory not found at {eval_path}. "
            "Update CASIA_EVAL_DIR_PLACEHOLDER / --eval-dir once the USB drive is mounted."
        )

    found_any = False
    summary = []
    for bin_stem, dataset_source in BIN_TO_DATASET_SOURCE.items():
        bin_path = eval_path / f"{bin_stem}.bin"
        if not bin_path.is_file():
            logger.warning(f"{bin_path.name} not found under {eval_path}, skipping.")
            continue
        found_any = True
        out_dir = output_path / dataset_source
        summary.append(extract_bin_file(bin_path, out_dir, dataset_source))

    if not found_any:
        raise FileNotFoundError(
            f"No known .bin files ({list(BIN_TO_DATASET_SOURCE)}) found under {eval_path}."
        )

    total_written = sum(s["written"] for s in summary)
    logger.info(f"Done. {total_written} total images extracted across {len(summary)} bin file(s).")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract images from CASIA-WebFace package eval/*.bin verification files."
    )
    parser.add_argument(
        "--eval-dir", type=str, default=CASIA_EVAL_DIR_PLACEHOLDER,
        help="Path to the CASIA package's eval/ folder (contains lfw.bin, agedb_30.bin, ...). "
             "PLACEHOLDER -- point this at the real path once available.",
    )
    parser.add_argument(
        "--output-root", type=str, default=CASIA_OUTPUT_ROOT_PLACEHOLDER,
        help="Root folder to write CASIA-<name>/ subfolders into. Defaults to the same "
             "clean_dataset root the rest of the pipeline reads from.",
    )
    args = parser.parse_args()
    extract_all(args.eval_dir, args.output_root)


if __name__ == "__main__":
    main()
