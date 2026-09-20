"""
pkface_test.py
===============
Evaluate PK-face test set against a Milvus collection.

For each test image:
  1. Extracts identity from filename (e.g. pa16_1949_3.jpg -> pa16_1949)
  2. Embeds the image
  3. Searches Milvus for top-K matches
  4. Checks if the correct identity is in the results

Outputs:
  - results.csv  (per-query details)
  - evaluation.md (accuracy report)

Usage:
    python pkface_test.py --test-folder D:\\PK-face_split\\test --collection pkface_train
    python pkface_test.py --test-folder /data/test --collection my_faces --top-k 10
    python pkface_test.py --test-folder /data/test --collection faces --limit 100
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from embedder import get_embedder, NoFaceDetectedError
from milvus_client import VectorDBClient, DEFAULT_URI, DEFAULT_COLLECTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pkface_test")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def extract_identity(filename: str) -> str:
    """Extract identity from PK-face filename.

    Pattern: {prefix}_{id}_{index}.jpg -> {prefix}_{id}
    Example: pa16_1949_3.jpg -> pa16_1949
    """
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0]
    return stem


def collect_test_images(test_folder: Path) -> List[Path]:
    """Collect all image files in the test folder."""
    return sorted(
        p for p in test_folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def run_evaluation(
    test_folder: Path,
    collection_name: str,
    milvus_uri: str = DEFAULT_URI,
    top_k: int = 5,
    limit: Optional[int] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, object]:
    images = collect_test_images(test_folder)
    if limit:
        images = images[:limit]

    if not images:
        raise ValueError(f"No images found in {test_folder}")

    logger.info(f"Found {len(images)} test images in {test_folder}")

    # Connect to Milvus
    db = VectorDBClient(uri=milvus_uri, collection_name=collection_name)
    db.connect()
    if not db.client.has_collection(collection_name):
        raise RuntimeError(
            f"Collection '{collection_name}' does not exist. "
            f"Run ingest_folder.py first to create it.")
    db.load_collection()
    logger.info(f"Collection '{collection_name}' has {db.count()} rows")

    embedder = get_embedder()

    results: List[dict] = []
    t0 = time.time()

    for img_path in tqdm(images, desc="Evaluating"):
        expected_identity = extract_identity(img_path.name)

        try:
            emb = embedder.embed_image(str(img_path)).embedding
        except NoFaceDetectedError:
            results.append({
                "query_file": img_path.name,
                "expected_identity": expected_identity,
                "top1_identity": None,
                "top1_score": None,
                "best_correct_score": None,
                "hit@1": False,
                "hit@5": False,
                "error": "no_face_detected",
            })
            continue
        except Exception as e:
            results.append({
                "query_file": img_path.name,
                "expected_identity": expected_identity,
                "top1_identity": None,
                "top1_score": None,
                "best_correct_score": None,
                "hit@1": False,
                "hit@5": False,
                "error": str(e),
            })
            continue

        hits = db.search(emb, top_k=top_k)

        # Extract identities from hit paths
        hit_identities = []
        for h in hits:
            hit_path = Path(h.image_path)
            hit_identities.append(extract_identity(hit_path.name))

        scores = [h.score for h in hits]

        # Find correct matches
        correct_scores = [
            s for ident, s in zip(hit_identities, scores)
            if ident == expected_identity
        ]

        top1 = hit_identities[0] if hit_identities else None

        results.append({
            "query_file": img_path.name,
            "expected_identity": expected_identity,
            "top1_identity": top1,
            "top1_score": scores[0] if scores else None,
            "best_correct_score": max(correct_scores) if correct_scores else None,
            "hit@1": bool(top1 == expected_identity),
            "hit@5": any(ident == expected_identity for ident in hit_identities),
            "error": None,
        })

    elapsed = time.time() - t0

    # Compute summary
    scored = [r for r in results if r["error"] is None]
    n_total = len(results)
    n_scored = len(scored)
    n_errors = n_total - n_scored

    hit1 = sum(r["hit@1"] for r in scored) / n_scored if n_scored else 0
    hitk = sum(r["hit@5"] for r in scored) / n_scored if n_scored else 0

    correct_scores = [r["best_correct_score"] for r in scored if r["best_correct_score"] is not None]
    mean_correct = statistics.fmean(correct_scores) if correct_scores else None

    # Per-identity accuracy
    by_identity: Dict[str, List[dict]] = defaultdict(list)
    for r in scored:
        by_identity[r["expected_identity"]].append(r)

    per_identity = {}
    for ident, rows in sorted(by_identity.items()):
        per_identity[ident] = {
            "n_images": len(rows),
            "hit@1": sum(r["hit@1"] for r in rows) / len(rows),
            "hit@5": sum(r["hit@5"] for r in rows) / len(rows),
        }

    summary = {
        "overall": {
            "n_total": n_total,
            "n_scored": n_scored,
            "n_errors": n_errors,
            "n_identities": len(by_identity),
            "hit@1": hit1,
            "hit@k": hitk,
            "top_k": top_k,
            "mean_correct_cosine": mean_correct,
            "elapsed_seconds": elapsed,
        },
        "per_identity": per_identity,
    }

    # Save outputs
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_results(results, out_dir / "results.csv")
        _write_report(summary, out_dir / "evaluation.md")
        logger.info(f"Results saved to {out_dir}")

    _print_summary(summary)
    return summary


def _write_results(results: List[dict], path: Path) -> None:
    fields = ["query_file", "expected_identity", "top1_identity", "top1_score",
              "best_correct_score", "hit@1", "hit@5", "error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: ("" if r.get(k) is None else r[k]) for k in fields})


def _write_report(summary: Dict[str, object], path: Path) -> None:
    o = summary["overall"]
    lines = [
        "# PK-face Test Evaluation",
        "",
        f"- Test folder: (see command line)",
        f"- Collection: (see command line)",
        f"- Top-K: {o['top_k']}",
        f"- Total images: {o['n_total']}",
        f"- Scored: {o['n_scored']}",
        f"- Errors (no face detected): {o['n_errors']}",
        f"- Unique identities: {o['n_identities']}",
        f"- **Hit@1: {o['hit@1']:.4f}**",
        f"- **Hit@{o['top_k']}: {o['hit@k']:.4f}**",
        f"- Mean cosine (correct matches): {_fmt(o['mean_correct_cosine'])}",
        f"- Time: {o['elapsed_seconds']:.1f}s",
        "",
        "## Per-identity accuracy",
        "",
        "| identity | n_images | hit@1 | hit@k |",
        "|---|---|---|---|",
    ]
    for ident, s in summary["per_identity"].items():
        lines.append(f"| {ident} | {s['n_images']} | {s['hit@1']:.4f} | {s['hit@5']:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _print_summary(summary: Dict[str, object]) -> None:
    o = summary["overall"]
    print("\n========== PK-face Evaluation ==========")
    print(f"  Total images    : {o['n_total']}")
    print(f"  Scored          : {o['n_scored']}")
    print(f"  Errors          : {o['n_errors']}")
    print(f"  Identities      : {o['n_identities']}")
    print(f"  Hit@1           : {o['hit@1']:.4f}")
    print(f"  Hit@{o['top_k']}           : {o['hit@k']:.4f}")
    print(f"  Mean correct    : {_fmt(o['mean_correct_cosine'])}")
    print(f"  Time            : {o['elapsed_seconds']:.1f}s")
    print("========================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PK-face test set against a Milvus collection.")
    parser.add_argument("--test-folder", type=str, required=True,
                        help="Path to PK-face test folder (flat, *.jpg).")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION,
                        help=f"Milvus collection name (default: {DEFAULT_COLLECTION}).")
    parser.add_argument("--milvus-uri", type=str, default=DEFAULT_URI,
                        help=f"Milvus URI (default: {DEFAULT_URI})")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top results to check (default: 5).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N test images (for quick testing).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for results.csv + evaluation.md "
                             "(default: <test-folder>_eval).")
    args = parser.parse_args()

    test_folder = Path(args.test_folder)
    if not test_folder.is_dir():
        print(f"Error: {test_folder} is not a directory.")
        return

    out_dir = Path(args.out) if args.out else test_folder.parent / f"{test_folder.name}_eval"

    run_evaluation(
        test_folder=test_folder,
        collection_name=args.collection,
        milvus_uri=args.milvus_uri,
        top_k=args.top_k,
        limit=args.limit,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
