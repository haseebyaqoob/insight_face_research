"""
extract_rec_images.py
=====================
Extract the aligned face images packed inside InsightFace-style RecordIO
training files (train.rec / train.idx / train.lst) back out as real .jpg
files organised per identity:

    <out>/<CASIA|VGG>/<identity_label>/0001.jpg
    <out>/<CASIA|VGG>/<identity_label>/0002.jpg
    ...

Designed for the CASIA-WebFace and VGG-Face aligned 112x112 sets stored
under D:\\Code\\insightface\\CASIA\\faces_webface_112x112 (and VGG\\...):

    CASIA/faces_webface_112x112/  train.rec  train.idx  train.lst  property
    VGG/faces_vgg_112x112/        train.rec  train.idx             property

Record format notes (mirrors D:\\Code\\insightface\\training_evaluation.py):
  - .idx is a text file mapping  record_index -> byte_offset  (one per line)
  - each record starts at its offset with a fixed-size header:
        CASIA: 24 bytes  (label is a float, always 0 here -> label taken
                          from train.lst when present)
        VGG:   32 bytes  (label is a uint16 at bytes 4..5, no .lst file)
  - the JPEG bytes follow the header; the record region runs up to the next
    record's offset (or EOF), padded to alignment, so the image bytes are
    sliced exactly from SOI (0xFFD8) to EOI (0xFFD9).

The extracted .jpg files are the ORIGINAL record bytes (no re-encode), so
they are identical to what the record file stored. The crops are pre-
aligned 112x112 ArcFace crops -> they can later be consumed with
is_aligned=True by merge_dedup_split.py / ingest_merged_split.py.

Usage (Windows venv python, from this repo):
    # extract EVERYTHING from both datasets
    python extract_rec_images.py --dataset-path D:\\Code\\insightface\\CASIA \\
                                 --dataset-path D:\\Code\\insightface\\VGG

    # quick smoke: only 500 images per dataset path
    python extract_rec_images.py --dataset-path D:\\Code\\insightface\\CASIA \\
                                 --dataset-path D:\\Code\\insightface\\VGG \\
                                 --max-images 500 --out extracted_smoke
"""

from __future__ import annotations

import argparse
import logging
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_rec_images")


def _resolve_dataset_dir(dataset_path: Path) -> Path:
    """Accept either the folder that CONTAINS train.rec, or its parent."""
    if (dataset_path / "train.rec").is_file():
        return dataset_path
    for child in sorted(dataset_path.iterdir()) if dataset_path.is_dir() else []:
        if child.is_dir() and (child / "train.rec").is_file():
            return child
    raise FileNotFoundError(
        f"No train.rec found under {dataset_path} (looked directly and one "
        f"level down). Expected e.g. D:/Code/insightface/CASIA/faces_webface_112x112"
    )


def _read_offsets(idx_path: Path) -> Dict[int, int]:
    """record_index -> byte_offset from the text .idx file."""
    offsets: Dict[int, int] = {}
    if not idx_path.is_file():
        raise FileNotFoundError(f"Index file not found: {idx_path}")
    with open(idx_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    offsets[int(parts[0])] = int(parts[1])
                except ValueError:
                    continue
    if not offsets:
        raise ValueError(f"Index file {idx_path} contained no usable entries.")
    return offsets


def _read_lst_labels(lst_path: Path) -> List[int]:
    """Return ordered list of identity labels from train.lst, one per line.

    CASIA .lst files have index=0 in the first column for every entry,
    so we return labels in file order for positional matching. The identity
    is taken from the path's second-to-last segment when the label column
    is 0.
    """
    labels: List[int] = []
    if not lst_path.is_file():
        return labels
    with open(lst_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            try:
                label = int(fields[2])
                path = fields[1]
                if label == 0 and "/" in path:
                    segments = path.rstrip("/").split("/")
                    if len(segments) >= 2:
                        label = int(segments[-2])
            except (ValueError, IndexError):
                continue
            labels.append(label)
    return labels


def _parse_header(raw: bytes, is_vgg: bool) -> Optional[dict]:
    if is_vgg:
        if len(raw) < 32:
            return None
        # VGG: magic(4) + label_uint16(2) + zeros(10) + idx_uint32(4) + zeros(12) = 32 bytes
        return {"label": struct.unpack_from("<H", raw, 4)[0],
                "id": struct.unpack_from("<I", raw, 16)[0]}
    # CASIA: flag(4) + label_float(4) + id_uint64(8) + reserved(4) = 24 bytes
    if len(raw) < 24:
        return None
    return {"label": int(struct.unpack_from("<f", raw, 4)[0]),
            "id": struct.unpack_from("<Q", raw, 8)[0]}


def _find_jpeg_bytes(region: bytes) -> Optional[bytes]:
    """Slice the exact original JPEG bytes (SOI..EOI) out of a record region."""
    start = region.find(b"\xff\xd8")
    if start < 0:
        return None
    end = region.find(b"\xff\xd9", start)
    if end < 0:
        return None
    jpg = region[start:end + 2]
    return jpg if len(jpg) > 100 else None


def extract_dataset(
    dataset_dir: Path,
    out_root: Path,
    max_images: int,
) -> Tuple[int, int]:
    """Extract images from one dataset's .rec/.idx/.lst into
    <out_root>/<parent-dir-name>/<seq>.jpg (flat). Returns
    (images_written, failures)."""
    rec_path = dataset_dir / "train.rec"
    idx_path = dataset_dir / "train.idx"
    lst_path = dataset_dir / "train.lst"

    offsets = _read_offsets(idx_path)
    lst_labels = _read_lst_labels(lst_path)
    has_lst = bool(lst_labels)
    is_vgg = not has_lst
    header_size = 32 if is_vgg else 24

    dest_parent = out_root / dataset_dir.parent.name
    dest_parent.mkdir(parents=True, exist_ok=True)

    sorted_offsets = sorted(offsets.items(), key=lambda kv: kv[1])
    written = 0
    failed = 0
    t0 = time.time()

    with open(rec_path, "rb") as f:
        fsize = rec_path.stat().st_size
        for pos, (idx, offset) in enumerate(sorted_offsets):
            if max_images and written >= max_images:
                break

            next_offset = (sorted_offsets[pos + 1][1]
                           if pos + 1 < len(sorted_offsets) else fsize)
            f.seek(offset)
            region = f.read(next_offset - offset)

            header = _parse_header(region[:max(32, header_size)], is_vgg)
            if header is None:
                failed += 1
                continue

            # --- KEY FIX: positional matching like training_evaluation.py ---
            label = header["label"]  # default: from record header
            if has_lst and pos < len(lst_labels):
                label = lst_labels[pos]  # positional match from .lst
            # --- END FIX ---

            jpg = _find_jpeg_bytes(region[header_size:])
            if jpg is None:
                failed += 1
                logger.warning(f"record idx={idx}: no JPEG payload found; skipped")
                continue

            out_file = dest_parent / f"{written + 1:05d}.jpg"
            out_file.write_bytes(jpg)
            written += 1

    logger.info(f"{dataset_dir.parent.name}: {written} images, {failed} failures "
                f"-> {dest_parent} ({time.time() - t0:.1f}s)")
    return written, failed


def main():
    parser = argparse.ArgumentParser(
        description="Extract .jpg files from CASIA/VGG RecordIO training sets "
                    "(train.rec/.idx/.lst). Output: flat folder per dataset.")
    parser.add_argument("--dataset-path", action="append", required=True, dest="dataset_paths",
                        help="Path to a dataset folder containing train.rec (repeatable: "
                             "pass both CASIA and VGG). Also accepts the parent folder "
                             "(e.g. D:\\Code\\insightface\\CASIA).")
    parser.add_argument("--out", type=str, default="extracted",
                        help="Output root. Each dataset is written to "
                             "<out>/<parent-dir-name>/*.jpg "
                             "(default: ./extracted).")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Stop after this many images PER dataset path "
                             "(0 = extract everything).")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    total_written = total_failed = 0
    for raw in args.dataset_paths:
        dataset_dir = _resolve_dataset_dir(Path(raw))
        logger.info(f"Extracting from {dataset_dir} ...")
        written, failed = extract_dataset(dataset_dir, out_root,
                                          args.max_images if args.max_images else 0)
        total_written += written
        total_failed += failed

    logger.info(f"All done: {total_written} images written to {out_root} "
                f"({total_failed} failures).")


if __name__ == "__main__":
    main()
