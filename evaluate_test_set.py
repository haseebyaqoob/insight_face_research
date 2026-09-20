"""
evaluate_test_set.py
=====================
Stage C: evaluate the TEST split of a merge_dedup_split.py run against the
train split stored in Milvus (see ingest_merged_split.py).

How the split maps to a recognition test
----------------------------------------
merge_dedup_split.py assigned one integer cluster_id per identity and split
every identity's photos as: 1 representative -> test/, the other distinct
photos -> train/ (byte-identical copies were dropped). So the ground-truth
person label of every test image is `cluster_<cluster_id>` from
manifest.csv, and that same label exists on its identity's train rows in
Milvus (stored as person_id by ingest_merged_split.py).

For each test image we:
    1. embed it with the SAME embedder (embed_image(is_aligned=True))
    2. search the merged train collection (top_k)
    3. score hit@1 / hit@5 = does the top-1 / any-of-top-k result carry the
       query's own person_id?

Identities whose cluster has NO train row at all (byte-identical-only
clusters: 1 test image + only dropped copies, nothing in train) are counted
and excluded from accuracy -- by construction there is nothing to match.

Outputs: <out>/results.csv (per query) + <out>/evaluation.md + a printed
summary table (overall and per original dataset_source).

Reuses (never re-implements):
    - embedder.get_embedder().embed_image(is_aligned=True)
    - milvus_client.VectorDBClient.search()
    - ingest_merged_split's manifest loader + DEFAULT_COLLECTION so the
      collection name can never drift between ingest and eval.

Usage (Windows venv python, repo root):
    python ingest_merged_split.py  --split-root D:\\Code\\embedding_model\\CASIA_smoke1
    python evaluate_test_set.py    --split-root D:\\Code\\embedding_model\\CASIA_smoke1
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from embedder import get_embedder
from milvus_client import VectorDBClient, DEFAULT_URI
from ingest_merged_split import (
    DEFAULT_COLLECTION,
    cluster_person_id,
    load_manifest_rows,
    missing_files,
    split_rows,
    train_cluster_ids,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_test_set")

RESULT_FIELDS = ["query_file", "query_dataset", "expected_cluster",
                 "top1_cluster", "top1_score", "best_correct_score",
                 "hit@1", "hit@5", "has_train_row"]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def no_train_clusters(test_rows: List[dict], train_rows: List[dict]) -> set:
    """cluster_ids present in test but absent from train (nothing to match)."""
    return {r["cluster_id"] for r in test_rows} - train_cluster_ids(train_rows)


def dest_overlap(train_rows: List[dict], test_rows: List[dict]) -> set:
    """Any dest_name that appears in BOTH splits (should never happen: the
    merge guarantees a copied file lives in exactly one split)."""
    train_dests = {r["dest_name"] for r in train_rows if r.get("dest_name")}
    return {r["dest_name"] for r in test_rows if r["dest_name"] in train_dests}


def summarize_results(results: List[dict]) -> Dict[str, object]:
    """Aggregate per-query rows into overall + per-dataset stats.
    Queries with has_train_row=False are excluded from accuracy."""
    scored = [r for r in results if r["has_train_row"]]
    n = len(scored)
    n_nontrain = len(results) - n

    overall = {
        "n_queries_total": len(results),
        "n_queries_no_train": n_nontrain,
        "n_queries_scored": n,
        "hit@1": (sum(r["hit@1"] for r in scored) / n) if n else None,
        "hit@5": (sum(r["hit@5"] for r in scored) / n) if n else None,
    }
    correct_scores = [r["best_correct_score"] for r in scored
                      if r["best_correct_score"] is not None]
    overall["n_correct"] = len(correct_scores)
    overall["mean_correct_cosine"] = (
        statistics.fmean(correct_scores) if correct_scores else None
    )

    per_dataset = {}
    by_ds: Dict[str, List[dict]] = defaultdict(list)
    for r in scored:
        by_ds[r["query_dataset"]].append(r)
    for ds, rows in sorted(by_ds.items()):
        per_dataset[ds] = {
            "n": len(rows),
            "hit@1": sum(x["hit@1"] for x in rows) / len(rows),
            "hit@5": sum(x["hit@5"] for x in rows) / len(rows),
        }
    return {"overall": overall, "per_dataset": per_dataset}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_eval(
    split_root: Path,
    milvus_uri: str = DEFAULT_URI,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 5,
    limit: Optional[int] = None,
    out: Optional[Path] = None,
) -> Dict[str, object]:
    all_rows = load_manifest_rows(split_root)
    train_rows = split_rows(all_rows, "train")
    test_rows = split_rows(all_rows, "test")

    if not test_rows:
        raise ValueError(f"0 test rows in {split_root}/manifest.csv.")
    if limit:
        test_rows = test_rows[:limit]

    overlap = dest_overlap(train_rows, test_rows)
    if overlap:
        raise ValueError(f"{len(overlap)} file(s) appear in BOTH train and test "
                         f"(first: {sorted(overlap)[0]}) -- split is corrupted.")

    missing = missing_files(split_root, test_rows)
    if missing:
        raise FileNotFoundError(f"{len(missing)} test file(s) missing on disk "
                                f"(first: {missing[0]}). Re-run merge_dedup_split.py.")

    untrained = no_train_clusters(test_rows, train_rows)
    if untrained:
        logger.warning(f"{len(untrained)} test identity cluster(s) have no train row "
                       f"(byte-identical-only clusters) -- they will be counted and "
                       f"excluded from accuracy.")

    db = VectorDBClient(uri=milvus_uri, collection_name=collection_name)
    db.connect()
    if not db.client.has_collection(collection_name):
        raise RuntimeError(
            f"Collection '{collection_name}' does not exist -- run "
            f"ingest_merged_split.py --collection-name {collection_name} first.")
    db.load_collection()
    logger.info(f"Collection '{collection_name}' has {db.count()} rows.")

    embedder = get_embedder()
    results: List[dict] = []
    for row in tqdm(test_rows, desc="Evaluating test split"):
        query_file = str(split_root / row["dest_name"])
        expected = cluster_person_id(row["cluster_id"])
        try:
            emb = embedder.embed_image(query_file, is_aligned=True).embedding
        except Exception as e:  # noqa: BLE001
            logger.warning(f"query embed failed for {query_file}: {e}")
            results.append({
                "query_file": query_file, "query_dataset": row["dataset_source"],
                "expected_cluster": row["cluster_id"],
                "top1_cluster": None, "top1_score": None, "best_correct_score": None,
                "hit@1": False, "hit@5": False,
                "has_train_row": row["cluster_id"] not in untrained,
            })
            continue

        hits = db.search(emb, top_k=top_k)
        hit_clusters = [h.person_id for h in hits]          # person_id="cluster_N"
        scores = [h.score for h in hits]
        correct_scores = [s for c, s in zip(hit_clusters, scores) if c == expected]
        top1 = hit_clusters[0] if hit_clusters else None
        results.append({
            "query_file": query_file,
            "query_dataset": row["dataset_source"],
            "expected_cluster": row["cluster_id"],
            "top1_cluster": top1.replace("cluster_", "") if top1 else None,
            "top1_score": scores[0] if scores else None,
            "best_correct_score": max(correct_scores) if correct_scores else None,
            "hit@1": bool(top1 == expected),
            "hit@5": any(c == expected for c in hit_clusters),
            "has_train_row": row["cluster_id"] not in untrained,
        })

    summary = summarize_results(results)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        _write_results(results, out / "results.csv")
        _write_report(summary, out / "evaluation.md", top_k)
    _print_summary(summary, top_k)
    return summary


def _write_results(results: List[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: ("" if r.get(k) is None else r[k]) for k in RESULT_FIELDS})


def _write_report(summary: Dict[str, object], path: Path, top_k: int) -> None:
    o = summary["overall"]
    lines = [
        "# Test-set evaluation vs merged train (Milvus)",
        "",
        f"- queries total: {o['n_queries_total']}",
        f"- queries with no train row (excluded): {o['n_queries_no_train']}",
        f"- scored: {o['n_queries_scored']}",
        f"- hit@1: {_fmt(o['hit@1'])}",
        f"- hit@{top_k}: {_fmt(o['hit@5'])}",
        f"- correct matches: {o['n_correct']}",
        f"- mean cosine of correct matches: {_fmt(o['mean_correct_cosine'])}",
        "",
        "## Per-dataset accuracy",
        "",
        "| dataset_source | n | hit@1 | hit@5 |",
        "|---|---|---|---|",
    ]
    for ds, s in summary["per_dataset"].items():
        lines.append(f"| {ds} | {s['n']} | {_fmt(s['hit@1'])} | {_fmt(s['hit@5'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _print_summary(summary: Dict[str, object], top_k: int) -> None:
    o = summary["overall"]
    print("\n========== Evaluation summary ==========")
    print(f"  queries total       : {o['n_queries_total']}")
    print(f"  no train row (excl) : {o['n_queries_no_train']}")
    print(f"  scored              : {o['n_queries_scored']}")
    print(f"  hit@1               : {_fmt(o['hit@1'])}")
    print(f"  hit@{top_k}               : {_fmt(o['hit@5'])}")
    print(f"  mean correct cosine : {_fmt(o['mean_correct_cosine'])}")
    print("\n  per-dataset hit@1:")
    for ds, s in summary["per_dataset"].items():
        print(f"    {ds:<20} n={s['n']:<5} hit@1={_fmt(s['hit@1'])}  hit@{top_k}={_fmt(s['hit@5'])}")
    print("========================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the merged TEST split against the train collection in Milvus.")
    parser.add_argument("--split-root", type=str, required=True,
                        help="Folder with manifest.csv + train/ + test/ "
                             "(output of merge_dedup_split.py).")
    parser.add_argument("--collection-name", type=str, default=DEFAULT_COLLECTION,
                        help=f"Collection populated by ingest_merged_split.py "
                             f"(default: {DEFAULT_COLLECTION}).")
    parser.add_argument("--milvus-uri", type=str, default=DEFAULT_URI)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N test images (smoke testing).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output folder for results.csv + evaluation.md "
                             "(default: <split-root>_eval).")
    args = parser.parse_args()

    run_eval(
        split_root=Path(args.split_root),
        milvus_uri=args.milvus_uri,
        collection_name=args.collection_name,
        top_k=args.top_k,
        limit=args.limit,
        out=Path(args.out) if args.out else None,
    )


if __name__ == "__main__":
    main()
