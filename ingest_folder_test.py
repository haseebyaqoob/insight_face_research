"""
ingest_folder_test.py
=====================
General-purpose script to ingest a folder of images into a Milvus collection,
with optional cache-first hybrid embedding (reuses vectors from a prior
merge_dedup_split.py run when available).

  - If the collection exists: inserts into it (skips already-indexed images).
  - If the collection doesn't exist: creates it first, then inserts.

Cache behavior:
    When the input folder contains a manifest.csv + _cache/ (i.e. it is the
    output of merge_dedup_split.py), the script reuses the cached embeddings
    from _cache/embeddings.npy (keyed by (dataset_source, name) from the
    manifest). Only cache-miss rows are embedded fresh with the model. The
    ONNX model is loaded lazily on the first miss.

    When no manifest or _cache/ is present, every image is embedded from
    disk (legacy behavior, identical to ingest_folder.py).

Usage:
    python ingest_folder_test.py --folder /path/to/images --collection my_faces
    python ingest_folder_test.py --folder D:\\Code\\embedding_model\\CASIA_split\\train \\
        --collection my_faces
    python ingest_folder_test.py --folder /data/images --collection faces --batch-size 64
    python ingest_folder_test.py --folder /data/images --collection faces --drop-existing
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from embedder import (
    get_embedder,
    NoFaceDetectedError,
    MODEL_VERSION,
    EMBEDDING_VERSION,
    ALIGNMENT_VERSION,
)
from milvus_client import VectorDBClient, DEFAULT_URI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_folder_test")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(folder: Path) -> List[Path]:
    """Collect all image files in the folder (non-recursive)."""
    images = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return images


# ---------------------------------------------------------------------------
# Cache-backed vectors (same logic as ingest_merged_split.py)
# ---------------------------------------------------------------------------
def load_cache_index(cache_dir: Path) -> Tuple[Optional[dict], Optional[np.ndarray]]:
    """Map (dataset_source, name) -> row index in _cache/embeddings.npy, plus
    the array itself. Returns (None, None) when the cache is missing or
    unusable (corrupt JSON, keys/rows length mismatch, bad shape); the caller
    then embeds everything from disk."""
    meta_path = cache_dir / "embed_meta.json"
    npy_path = cache_dir / "embeddings.npy"
    if not (meta_path.is_file() and npy_path.is_file()):
        return None, None
    try:
        meta = json.loads(meta_path.read_text("utf-8"))
        keys = meta["keys"]
        arr = np.load(str(npy_path), mmap_mode="r")
        if arr.ndim != 2 or arr.shape[0] != len(keys) or arr.shape[1] < 1:
            return None, None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        return None, None
    return {tuple(k): i for i, k in enumerate(keys)}, arr


def load_manifest_keys(manifest_path: Path) -> List[Tuple[str, str]]:
    """Read manifest.csv and return list of (dataset_source, name) pairs
    for each row, in file order (matching embeddings.npy row order)."""
    if not manifest_path.is_file():
        return []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [(r["dataset_source"], r["name"]) for r in rows
            if r.get("dataset_source", "").strip() and r.get("name", "").strip()]


def build_path_to_key_map(
    manifest_keys: List[Tuple[str, str]],
) -> Dict[str, Tuple[str, str]]:
    """Build a mapping from image filename (without extension) to
    (dataset_source, name) cache key. Assumes files in the folder follow
    the merge_dedup_split naming convention: <dataset>_<name>.<ext>."""
    mapping: Dict[str, Tuple[str, str]] = {}
    for ds, name in manifest_keys:
        # The merge step names files as {dataset}_{name}.{ext}
        base = f"{ds}_{name}"
        mapping[base] = (ds, name)
    return mapping


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def run_ingest(
    folder: Path,
    collection_name: str,
    milvus_uri: str = DEFAULT_URI,
    batch_size: int = 32,
    drop_existing: bool = False,
    limit: Optional[int] = None,
    dataset_source: str = "",
    failures_log: str = "ingest_failures.log",
) -> Dict[str, int]:
    images = collect_images(folder)
    if limit:
        images = images[:limit]

    if not images:
        raise ValueError(f"No images found in {folder}")

    logger.info(f"Found {len(images)} images in {folder}")

    # --- Cache detection (looks in parent dir — merge output layout) -------
    cache_root = folder.parent
    manifest_path = cache_root / "manifest.csv"
    key_to_idx, arr = load_cache_index(cache_root / "_cache")
    path_to_key: Optional[Dict[str, Tuple[str, str]]] = None

    if arr is not None:
        manifest_keys = load_manifest_keys(manifest_path)
        if manifest_keys:
            path_to_key = build_path_to_key_map(manifest_keys)
            logger.info(
                f"Using cached embeddings ({arr.shape[0]} vectors, dim "
                f"{arr.shape[1]}); only cache-miss rows will be embedded."
            )
        else:
            logger.warning(
                f"Cache found at {cache_root / '_cache'} but no usable "
                f"manifest.csv at {manifest_path}; embedding all "
                f"{len(images)} image(s) with the model."
            )
            key_to_idx, arr = None, None
    else:
        if manifest_path.is_file():
            logger.warning(
                f"manifest.csv found at {manifest_path} but no usable "
                f"_cache/ at {cache_root / '_cache'}; embedding all "
                f"{len(images)} image(s) with the model."
            )

    # --- Embedder (lazy if cache is usable) --------------------------------
    if arr is None:
        embedder = get_embedder()
        sample = embedder.embed_image(str(images[0]))
        dim = len(sample.embedding)
    else:
        embedder = None  # lazy: loaded on first cache miss
        dim = int(arr.shape[1])

    logger.info(f"Embedding dimension: {dim}")

    # --- DB setup ----------------------------------------------------------
    db = VectorDBClient(uri=milvus_uri, collection_name=collection_name)
    db.connect()
    db.create_collection(dim=dim, drop_existing=drop_existing)
    db.load_collection()

    existing_count = db.count()
    if existing_count:
        logger.info(
            f"Collection '{collection_name}' has {existing_count} existing rows; "
            f"skipping already-indexed images (resume)."
        )

    # --- Per-row ingest loop -----------------------------------------------
    failures: List[str] = []
    batch: List[dict] = []
    n_inserted = n_skipped = n_failed = 0
    n_from_cache = n_embedded = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal n_inserted, n_failed
        if not batch:
            return
        try:
            db.insert_embeddings(batch)
            n_inserted += len(batch)
        except Exception as e:
            n_failed += len(batch)
            failures.extend(f"{row['image_path']}\tinsert_failed: {e}" for row in batch)
            logger.warning(f"Batch insert failed ({len(batch)} images): {e}")
        batch.clear()

    for img_path in tqdm(images, desc="Ingesting"):
        path_str = str(img_path)

        # Skip if already indexed
        try:
            if db.image_already_indexed(path_str):
                n_skipped += 1
                continue
        except Exception as e:
            logger.debug(f"Skip-check failed for {path_str}, embedding anyway: {e}")

        # --- Cache lookup --------------------------------------------------
        vector = None
        if arr is not None and path_to_key is not None:
            base = img_path.name  # full filename, e.g. CASIA_00001.jpg
            cache_key = path_to_key.get(base)
            if cache_key is not None:
                i = key_to_idx.get(cache_key)
                if i is not None:
                    cached = np.asarray(arr[i], dtype=np.float32).reshape(-1)
                    if np.linalg.norm(cached) > 1e-9:
                        vector = cached
                    else:
                        logger.warning(
                            f"Cached embedding is zero for {path_str}; re-embedding."
                        )

        # --- Embed on cache miss -------------------------------------------
        if vector is None:
            if embedder is None:
                embedder = get_embedder()
                logger.info("Loaded AntelopeV2 embedder for cache-miss row(s).")
            try:
                result = embedder.embed_image(path_str)
            except NoFaceDetectedError:
                n_failed += 1
                failures.append(f"{path_str}\tno_face_detected")
                continue
            except Exception as e:
                n_failed += 1
                failures.append(f"{path_str}\terror: {e}")
                continue
            vector = np.asarray(result.embedding, dtype=np.float32).reshape(-1)
            n_embedded += 1
        else:
            n_from_cache += 1

        # Determine dataset_source for this row
        row_dataset = dataset_source
        if path_to_key is not None:
            base = img_path.name
            cache_key = path_to_key.get(base)
            if cache_key is not None:
                row_dataset = cache_key[0]

        batch.append({
            "embedding": vector,
            "image_path": path_str,
            "dataset_source": row_dataset,
            "person_id": "",
            "model_version": MODEL_VERSION,
            "embedding_version": EMBEDDING_VERSION,
            "alignment_version": ALIGNMENT_VERSION,
        })

        if len(batch) >= batch_size:
            flush()

    flush()

    if failures:
        Path(failures_log).write_text("\n".join(failures) + "\n", encoding="utf-8")
        logger.warning(f"{len(failures)} image(s) failed -- see {failures_log}")

    elapsed = time.time() - t0
    stats = {
        "inserted": n_inserted, "skipped": n_skipped, "failed": n_failed,
        "from_cache": n_from_cache, "embedded": n_embedded,
    }
    logger.info(
        f"Done in {elapsed:.1f}s. Inserted={n_inserted}, "
        f"skipped(already indexed)={n_skipped}, failed={n_failed}, "
        f"from_cache={n_from_cache}, embedded_fresh={n_embedded}, "
        f"total rows in '{collection_name}'={db.count()}"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a folder of images into a Milvus collection "
                    "(with cache-first hybrid embedding when available).")
    parser.add_argument("--folder", type=str, required=True,
                        help="Path to folder containing images.")
    parser.add_argument("--collection", type=str, required=True,
                        help="Milvus collection name (created if missing).")
    parser.add_argument("--milvus-uri", type=str, default=DEFAULT_URI,
                        help=f"Milvus URI (default: {DEFAULT_URI})")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Insert batch size (default: 32).")
    parser.add_argument("--drop-existing", action="store_true",
                        help="Drop and recreate the collection if it exists.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of images to ingest (for testing).")
    parser.add_argument("--dataset-source", type=str, default="",
                        help="Dataset source tag for metadata (overridden by "
                             "manifest when cache is present).")
    parser.add_argument("--failures-log", type=str, default="ingest_failures.log",
                        help="Log file for failed images.")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory.")
        return

    run_ingest(
        folder=folder,
        collection_name=args.collection,
        milvus_uri=args.milvus_uri,
        batch_size=args.batch_size,
        drop_existing=args.drop_existing,
        limit=args.limit,
        dataset_source=args.dataset_source,
        failures_log=args.failures_log,
    )


if __name__ == "__main__":
    main()
