"""
ingest_merged_split.py
=======================
Stage B: ingest the TRAIN split produced by merge_dedup_split.py into a
fresh Milvus collection so test-set evaluation can run against it.

The merged split is FLAT -- train/CASIA-AgeDB-30_img_00042.jpg -- with the
original dataset in the FILENAME prefix and the real identity label
(cluster_id from manifest.csv) available only in the manifest.

Vector source (hybrid, one path):
    merge_dedup_split.py already embeds every image during clustering and
    caches the vectors at <split_root>/_cache/embeddings.npy with row keys
    (dataset_source, name) in <split_root>/_cache/embed_meta.json. This
    script therefore:
      1. skips rows already in the target collection (skip_existing, by
         image_path);
      2. otherwise reuses the cached vector when the row's
         (dataset_source, name) key is present and nonzero;
      3. otherwise embeds THAT ONE image fresh with the model
         (embedder.embed_image(path, is_aligned=True)) -- the AntelopeV2
         ONNX model is loaded lazily, only on the first cache miss, so a
         fully-cached ingest never loads it.
    If the whole _cache/ folder is missing or unusable it warns once and
    embeds every train image (legacy behavior).

Rows are stored with:
    image_path    = str(<split_root>/<dest_name>)   (the file actually embedded;
                    unique across the split, so re-runs skip cleanly)
    dataset_source= original dataset name from the manifest (e.g. CASIA-AgeDB-30)
    person_id     = "cluster_<cluster_id>"          (ground-truth identity label
                    linking test queries to their train gallery rows)

Collection: a NEW collection (default "face_embeddings_merged") so the old
"face_embeddings" collection (36k+ unmerged CASIA-* rows with empty
person_id) is never polluted.

Usage (Windows venv python, repo root):
    python ingest_merged_split.py --split-root D:\\Code\\embedding_model\\CASIA_split
    python ingest_merged_split.py --split-root D:\\Code\\embedding_model\\CASIA_split \
        --collection-name face_embeddings_merged --drop-existing   # fresh full run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from embedder import get_embedder, NoFaceDetectedError
from embedder import MODEL_VERSION, EMBEDDING_VERSION, ALIGNMENT_VERSION
from milvus_client import VectorDBClient, DEFAULT_URI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_merged_split")

# New, dedicated collection. The existing "face_embeddings" collection from
# earlier bootstraps is left untouched by default.
DEFAULT_COLLECTION = "face_embeddings_merged"

MANIFEST_COLUMNS = [
    "dataset_source", "name", "path", "md5", "cluster_id", "split", "dest_name",
]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested offline)
# ---------------------------------------------------------------------------
def load_manifest_rows(split_root: Path) -> List[dict]:
    """Read <split_root>/manifest.csv into a list of dicts. Raises loudly when
    the file or a required column is missing."""
    manifest_path = split_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest.csv under {split_root} -- did you run merge_dedup_split.py "
            f"with --out {split_root} first?"
        )
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in MANIFEST_COLUMNS if not rows or c not in rows[0]]
    if missing:
        raise ValueError(f"manifest.csv is missing required column(s): {missing}")
    return rows


def split_rows(rows: List[dict], split_name: str) -> List[dict]:
    """Rows whose split == split_name AND that have a real dest_name (i.e.
    rows that were actually copied to disk by the merge step)."""
    return [r for r in rows
            if r.get("split") == split_name and r.get("dest_name", "").strip()]


def missing_files(split_root: Path, rows: List[dict]) -> List[str]:
    """dest_names under split_root that do not exist on disk."""
    return [r["dest_name"] for r in rows
            if not (split_root / r["dest_name"]).is_file()]


def cluster_person_id(cluster_id: str) -> str:
    return f"cluster_{cluster_id}"


def train_cluster_ids(rows: List[dict]) -> set:
    """cluster_ids that appear in the train split (i.e. are searchable)."""
    return {r["cluster_id"] for r in split_rows(rows, "train")}


# ---------------------------------------------------------------------------
# Cache-backed vectors
# ---------------------------------------------------------------------------
def load_cache_index(cache_dir: Path):
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


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def run_ingest(
    split_root: Path,
    milvus_uri: str = DEFAULT_URI,
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = 32,
    skip_existing: bool = True,
    drop_existing: bool = False,
    limit: Optional[int] = None,
    failures_log: str = "ingest_failures.log",
) -> Dict[str, int]:
    rows = split_rows(load_manifest_rows(split_root), "train")
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError(
            f"0 train rows in {split_root}/manifest.csv. If CASIA_split/train is empty, "
            f"finish the merge first: merge_dedup_split.py --root CASIA --out {split_root}"
        )

    missing = missing_files(split_root, rows)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} train file(s) referenced by the manifest are missing "
            f"on disk (first: {missing[0]}). Re-run merge_dedup_split.py."
        )

    logger.info(f"Ingesting {len(rows)} train image(s) from {split_root} ...")

    key_to_idx, arr = load_cache_index(split_root / "_cache")
    if arr is None:
        logger.warning(f"No usable embedding cache at {split_root / '_cache'}; "
                       f"embedding all {len(rows)} train image(s) with the model.")
        embedder = get_embedder()
        dim = embedder.embedding_dim
    else:
        logger.info(f"Using cached embeddings ({arr.shape[0]} vectors, dim "
                    f"{arr.shape[1]}); only cache-miss rows will be embedded.")
        embedder = None                       # lazy: loaded on the first cache miss
        dim = int(arr.shape[1])

    db = VectorDBClient(uri=milvus_uri, collection_name=collection_name)
    db.connect()
    db.create_collection(dim=dim, drop_existing=drop_existing)
    db.load_collection()
    existing = db.count()
    if existing:
        logger.info(f"Collection '{collection_name}' already has {existing} row(s); "
                    f"skipping already-indexed images (resume).")

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
        except Exception as e:  # noqa: BLE001
            n_failed += len(batch)
            failures.extend(f"{row['image_path']}\tinsert_failed: {e}" for row in batch)
            logger.warning(f"Batch insert failed ({len(batch)} images): {e}")
        batch.clear()

    for row in tqdm(rows, desc="Ingesting train split"):
        path_str = str(split_root / row["dest_name"])

        if skip_existing:
            try:
                if db.image_already_indexed(path_str):
                    n_skipped += 1
                    continue
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Skip-check failed for {path_str}, embedding anyway: {e}")

        vector = None
        if arr is not None:
            i = key_to_idx.get((row["dataset_source"], row["name"]))
            if i is not None:
                cached = np.asarray(arr[i], dtype=np.float32).reshape(-1)
                if np.linalg.norm(cached) > 1e-9:
                    vector = cached
                else:
                    logger.warning(f"Cached embedding is zero for {path_str}; "
                                   f"re-embedding.")

        if vector is None:                     # cache miss -> embed fresh
            if embedder is None:               # lazy model load
                embedder = get_embedder()
                logger.info("Loaded AntelopeV2 embedder for cache-miss row(s).")
            try:
                result = embedder.embed_image(path_str, is_aligned=True)
            except NoFaceDetectedError:
                n_failed += 1
                failures.append(f"{path_str}\tno_face_detected")
                continue
            except Exception as e:  # noqa: BLE001
                n_failed += 1
                failures.append(f"{path_str}\terror: {e}")
                continue
            vector = np.asarray(result.embedding, dtype=np.float32).reshape(-1)
            n_embedded += 1
        else:
            n_from_cache += 1

        batch.append({
            "embedding": vector,
            "image_path": path_str,
            "dataset_source": row["dataset_source"],
            "person_id": cluster_person_id(row["cluster_id"]),
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
    stats = {"inserted": n_inserted, "skipped": n_skipped, "failed": n_failed,
             "from_cache": n_from_cache, "embedded": n_embedded}
    logger.info(
        f"Done in {elapsed:.1f}s. Inserted={n_inserted}, "
        f"skipped(already indexed)={n_skipped}, failed={n_failed}, "
        f"from_cache={n_from_cache}, embedded_fresh={n_embedded}, "
        f"total rows in '{collection_name}'={db.count()}"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Ingest the merged TRAIN split into a fresh Milvus collection.")
    parser.add_argument("--split-root", type=str, required=True,
                        help="Folder with manifest.csv + train/<dataset>_<name>.jpg "
                             "(output of merge_dedup_split.py).")
    parser.add_argument("--milvus-uri", type=str, default=DEFAULT_URI)
    parser.add_argument("--collection-name", type=str, default=DEFAULT_COLLECTION,
                        help=f"Fresh collection (default: {DEFAULT_COLLECTION}); "
                             "the old face_embeddings collection is never touched.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                        help="Ingest only the first N train rows (smoke testing).")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Re-embed and re-insert even if already indexed.")
    parser.add_argument("--drop-existing", action="store_true",
                        help="Drop the target collection first (fresh start).")
    parser.add_argument("--failures-log", type=str, default="ingest_failures.log")
    args = parser.parse_args()

    stats = run_ingest(
        split_root=Path(args.split_root),
        milvus_uri=args.milvus_uri,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip_existing,
        drop_existing=args.drop_existing,
        limit=args.limit,
        failures_log=args.failures_log,
    )
    print(f"ingest summary: {stats}")


if __name__ == "__main__":
    main()
