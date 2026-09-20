"""
ingest_folder.py
=================
General-purpose script to ingest a folder of images into a Milvus collection.

  - If the collection exists: inserts into it (skips already-indexed images).
  - If the collection doesn't exist: creates it first, then inserts.

Usage:
    python ingest_folder.py --folder /path/to/images --collection my_faces
    python ingest_folder.py --folder D:\\PK-face_split\\train --collection pkface_train
    python ingest_folder.py --folder /data/images --collection faces --batch-size 64
    python ingest_folder.py --folder /data/images --collection faces --drop-existing
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

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
logger = logging.getLogger("ingest_folder")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(folder: Path) -> List[Path]:
    """Collect all image files in the folder (non-recursive)."""
    images = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return images


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

    db = VectorDBClient(uri=milvus_uri, collection_name=collection_name)
    db.connect()

    # Determine embedding dimension from the embedder
    embedder = get_embedder()
    sample = embedder.embed_image(str(images[0]))
    dim = len(sample.embedding)
    logger.info(f"Embedding dimension: {dim}")

    # Create collection (or reuse if exists)
    db.create_collection(dim=dim, drop_existing=drop_existing)
    db.load_collection()

    existing_count = db.count()
    logger.info(f"Collection '{collection_name}' has {existing_count} existing rows")

    failures: List[str] = []
    batch: List[dict] = []
    n_inserted = 0
    n_skipped = 0
    n_failed = 0
    t0 = time.time()

    def flush():
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

        # Embed
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

        batch.append({
            "embedding": result.embedding,
            "image_path": path_str,
            "dataset_source": dataset_source,
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
    stats = {"inserted": n_inserted, "skipped": n_skipped, "failed": n_failed}
    logger.info(
        f"Done in {elapsed:.1f}s. Inserted={n_inserted}, "
        f"skipped(already indexed)={n_skipped}, failed={n_failed}, "
        f"total rows in '{collection_name}'={db.count()}"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a folder of images into a Milvus collection.")
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
                        help="Dataset source tag for metadata.")
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
