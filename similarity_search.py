"""
similarity_search.py
====================
Query a folder of images against a Milvus collection and produce a CSV
with the top-K similarity results per query image, plus a dataset-source
count summary.

For each image in the folder:
  1. Embed it (RetinaFace detection + AntelopeV2)
  2. Search Milvus for top-K cosine hits
  3. Record person_id, dataset_source, and score for every hit

Outputs:
  - <output>.csv   — one row per query, columns for each rank's
                      person_id, score, and dataset_source
  - Console summary with dataset-source distribution across all hits

Usage:
    python similarity_search.py --folder /path/to/query_images
    python similarity_search.py --folder D:\\queries --collection pkface_embedding --top-k 10
    python similarity_search.py --folder /data/queries --output my_results.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from embedder import get_embedder, NoFaceDetectedError
from milvus_client import VectorDBClient, DEFAULT_URI, DEFAULT_COLLECTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("similarity_search")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(folder: Path) -> List[Path]:
    """Collect all image files in the folder (non-recursive)."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def run_search(
    folder: Path,
    collection_name: str,
    milvus_uri: str = DEFAULT_URI,
    top_k: int = 5,
    output: Optional[Path] = None,
) -> Dict[str, object]:
    images = collect_images(folder)
    if not images:
        raise ValueError(f"No images found in {folder}")

    logger.info(f"Found {len(images)} query images in {folder}")

    db = VectorDBClient(uri=milvus_uri, collection_name=collection_name)
    db.connect()
    if not db.client.has_collection(collection_name):
        raise RuntimeError(
            f"Collection '{collection_name}' does not exist. "
            f"Run ingest_folder.py or ingest_merged_split.py first.")
    db.load_collection()
    logger.info(f"Collection '{collection_name}' has {db.count()} rows")

    embedder = get_embedder()

    all_results: List[dict] = []
    dataset_counter: Counter = Counter()
    n_success = 0
    n_errors = 0
    t0 = time.time()

    for img_path in tqdm(images, desc="Searching"):
        try:
            emb = embedder.embed_image(str(img_path)).embedding
        except NoFaceDetectedError:
            logger.warning(f"No face detected in {img_path.name}, skipping.")
            n_errors += 1
            row = {"query_file": img_path.name, "error": "no_face_detected"}
            for rank in range(1, top_k + 1):
                row[f"rank{rank}_person"] = ""
                row[f"rank{rank}_score"] = ""
                row[f"rank{rank}_dataset"] = ""
            all_results.append(row)
            continue
        except Exception as e:
            logger.warning(f"Embedding failed for {img_path.name}: {e}")
            n_errors += 1
            row = {"query_file": img_path.name, "error": str(e)}
            for rank in range(1, top_k + 1):
                row[f"rank{rank}_person"] = ""
                row[f"rank{rank}_score"] = ""
                row[f"rank{rank}_dataset"] = ""
            all_results.append(row)
            continue

        hits = db.search(emb, top_k=top_k)
        n_success += 1

        row = {"query_file": img_path.name, "error": ""}
        for rank, hit in enumerate(hits, start=1):
            person = Path(hit.image_path).stem if hit.image_path else hit.person_id or "(unknown)"
            row[f"rank{rank}_person"] = person
            row[f"rank{rank}_score"] = f"{hit.score:.6f}"
            row[f"rank{rank}_dataset"] = hit.dataset_source
            dataset_counter[hit.dataset_source or "(unknown)"] += 1

        # Fill remaining ranks if fewer hits than top_k
        for rank in range(len(hits) + 1, top_k + 1):
            row[f"rank{rank}_person"] = ""
            row[f"rank{rank}_score"] = ""
            row[f"rank{rank}_dataset"] = ""

        all_results.append(row)

    elapsed = time.time() - t0

    # Build CSV field names
    fields = ["query_file", "error"]
    for rank in range(1, top_k + 1):
        fields.extend([f"rank{rank}_person", f"rank{rank}_score", f"rank{rank}_dataset"])

    # Write CSV
    if output is None:
        output = folder.parent / f"{folder.name}_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in fields})

    logger.info(f"Results saved to {output}")

    # Print summary
    _print_summary(len(images), n_success, n_errors, dataset_counter, top_k, elapsed)

    return {
        "total_queries": len(images),
        "success": n_success,
        "errors": n_errors,
        "dataset_counts": dict(dataset_counter),
        "output_csv": str(output),
    }


def _print_summary(
    total: int,
    success: int,
    errors: int,
    dataset_counter: Counter,
    top_k: int,
    elapsed: float,
) -> None:
    print("\n========== Similarity Search Results ==========")
    print(f"  Queries : {total}")
    print(f"  Success : {success}")
    print(f"  Errors  : {errors}")
    print(f"  Top-K   : {top_k}")
    print(f"  Time    : {elapsed:.1f}s")
    if dataset_counter:
        total_hits = sum(dataset_counter.values())
        print(f"\n  Dataset distribution in top-{top_k} hits ({total_hits} total):")
        for ds, count in dataset_counter.most_common():
            pct = count / total_hits * 100
            print(f"    {ds:<30} {count:>5}  ({pct:.1f}%)")
    print("================================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Query a folder of images against a Milvus collection "
                    "and output top-K similarity results as CSV.")
    parser.add_argument("--folder", type=str, required=True,
                        help="Path to folder containing query images.")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION,
                        help=f"Milvus collection name (default: {DEFAULT_COLLECTION}).")
    parser.add_argument("--milvus-uri", type=str, default=DEFAULT_URI,
                        help=f"Milvus URI (default: {DEFAULT_URI})")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top results per query (default: 5).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: <folder>_results.csv).")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory.")
        return

    output = Path(args.output) if args.output else None

    run_search(
        folder=folder,
        collection_name=args.collection,
        milvus_uri=args.milvus_uri,
        top_k=args.top_k,
        output=output,
    )


if __name__ == "__main__":
    main()
