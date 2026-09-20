"""
merge_dedup_split.py
====================
Merges multiple face-image datasets under a common root into ONE clean,
duplicate-free whole and splits it into train/ + test/ folders.

Duplicate definition (user requirement): "same person, any photo". The
source folders (e.g. the CASIA-* benchmark extracts under CASIA/) are flat
img_%05d.jpg sets with NO person names on disk, and different datasets
reuse the same people/photos (CALFW/CPLFW/CFP are built from LFW people),
so identity is recovered from the images themselves:

    1. MD5 of decoded pixels  -> exact-photo duplicates (cheap pass).
    2. AntelopeV2 embeddings  -> identity clustering by cosine similarity
       (thresholded connected components). Pixel-identical photos trivially
       co-cluster (identical pixels -> identical embedding); different
       photos of the same person co-cluster when similar enough.

Per identity group of N images: exactly ONE image goes to test/, the other
N-1 go to train/. Byte-identical copies of the SAME photo never survive
twice (a merge must be duplication-free): one copy may be used (as the
test representative or a train sample), the rest are counted as
dropped_exact_dup and are NOT copied. Identities that appear exactly once
(singletons -- never duplicated anywhere) stay in train by default so no
unique data is lost ("try to get as much data").

Every output filename carries its dataset provenance:
    train/CASIA-LFW_img_00042.jpg   test/CASIA-AgeDB-30_img_00123.jpg

Non-destructive: originals are never modified or deleted. Output goes to
--out (which must NOT be inside --root); every decision is logged in
<out>/manifest.csv and summarized in <out>/report.md. Resumable:
embeddings are cached to <out>/_cache/, so an interrupted run resumes.

Reuses (imports are side-effect free):
    - embedder.embed_image(is_aligned=True)   pre-aligned 112x112 crops
      (CASIA-* folders) skip RetinaFace detection.

Generic by design: every top-level folder under --root that contains image
files is one dataset, named after that folder. Run on the 6 CASIA-* folders
now; drop VGG / PK-Face folders under a root later and re-run (they get
picked up automatically). VGG-style raw (unaligned) sources would need
--no-aligned, which is surfaced but not part of this run.

Usage
-----
    # Full run on CASIA/ (all 6 datasets, ~36k images):
    python merge_dedup_split.py --root "D:\\Code\\embedding_model\\CASIA" \
        --out "D:\\Code\\embedding_model\\CASIA_split"

    # Smoke test on a small cross-dataset subset:
    python merge_dedup_split.py --root "D:\\Code\\embedding_model\\CASIA" \
        --out "D:\\Code\\embedding_model\\CASIA_smoke" --limit 300 --seed 1

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from embedder import embed_image

# --- Inlined from the former dedup helper module (no longer imported) ---
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Lower index = higher priority = kept as the representative when a duplicate
# group spans folders. Deliberately different from any "scan order" list --
# AgeDB-30 is highest priority for representative selection.
DEDUP_PRIORITY_FOLDERS = ["AgeDB-30", "LFW", "CALFW", "CPLFW", "CFP-FP"]


def hash_image(path: Path) -> str | None:
    """MD5 of decoded pixel bytes. None if the image can't be decoded."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    return hashlib.md5(img.tobytes()).hexdigest()
# --- end inline ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("merge_dedup_split")

EMBEDDING_DIM = 512           # AntelopeV2 (glintr100) output width
EMBED_BLOCK = 2048            # rows per cosine chunk (2048 x N x 4B ~ 300 MB @ N=36k)
EDGE_FLOOR = 0.25             # cosine pairs >= floor stored once; re-clustering is then free
SENSITIVITY_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]

MANIFEST_FIELDS = ["dataset_source", "name", "path", "md5", "cluster_id",
                   "split", "dest_name"]

# split values
SPLIT_TEST = "test"
SPLIT_TRAIN = "train"
SPLIT_DUP = "dropped_exact_dup"
SPLIT_SINGLETON = "dropped_singleton"
SPLIT_FAIL_DECODE = "failed_decode"
SPLIT_FAIL_EMBED = "failed_embed"


# ---------------------------------------------------------------------------
# Representative priority: lower tuple = kept first. Mirrors the validated
# ordering (AgeDB-30 > LFW > CALFW > CPLFW > CFP-FP), with
# the "CASIA-" prefix stripped so the same order applies to CASIA-* folders.
# Any unknown/new dataset (VGG, PK-Face, ...) sorts last.
# ---------------------------------------------------------------------------
def _priority_key(dataset_source: str) -> Tuple[int, str]:
    bare = dataset_source
    for prefix in ("CASIA-",):
        if dataset_source.startswith(prefix):
            bare = dataset_source[len(prefix):]
            break
    try:
        prio = DEDUP_PRIORITY_FOLDERS.index(bare)
    except ValueError:
        try:
            prio = DEDUP_PRIORITY_FOLDERS.index(dataset_source)
        except ValueError:
            prio = len(DEDUP_PRIORITY_FOLDERS)
    return (prio, dataset_source)


# ---------------------------------------------------------------------------
# Phase 0/1: discovery + exact-photo MD5 inventory
# ---------------------------------------------------------------------------
def discover_datasets(root: Path) -> List[str]:
    found: List[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if any(x.is_file() and x.suffix.lower() in IMAGE_EXTS for x in p.rglob("*")):
            found.append(p.name)
    return found


def _image_files(dataset_dir: Path) -> List[Path]:
    return sorted(p for p in dataset_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def collect_images(root: Path, datasets: List[str], limit: Optional[int]) -> List[dict]:
    """Deterministic flat record list. --limit takes a round-robin slice
    ACROSS datasets so a smoke test spans >= 2 folders (cross-dataset
    duplicates are the whole point)."""
    per_dataset = {d: _image_files(root / d) for d in datasets}
    records: List[dict] = []

    def make_rec(d: str, p: Path) -> dict:
        return {"dataset_source": d, "name": p.name, "path": str(p),
                "md5": None, "cluster_id": None, "split": None, "dest_name": None}

    if limit is None:
        for d in datasets:
            records.extend(make_rec(d, p) for p in per_dataset[d])
        return records

    iters = {d: iter(per_dataset[d]) for d in datasets}
    while len(records) < limit:
        progressed = False
        for d in datasets:
            if len(records) >= limit:
                break
            try:
                p = next(iters[d])
            except StopIteration:
                continue
            progressed = True
            records.append(make_rec(d, p))
        if not progressed:
            break
    return records


def _record_key(r: dict) -> Tuple[str, str]:
    return (r["dataset_source"], r["name"])


def run_md5_pass(records: List[dict], cache_dir: Path) -> None:
    """Exact-pixel MD5 (decode + hash) for every image. Cached to
    _cache/inventory.json and reused when the dataset/name list is identical,
    so re-runs don't re-decode 36k images."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    inv_path = cache_dir / "inventory.json"
    expected = [_record_key(r) for r in records]

    if inv_path.exists():
        try:
            saved = json.loads(inv_path.read_text("utf-8"))
            if saved.get("keys") == expected:
                by_key = {(e["dataset_source"], e["name"]): e for e in saved["records"]}
                for r in records:
                    old = by_key.get(_record_key(r))
                    if old is not None:
                        r["md5"] = old["md5"]
                        r["split"] = old.get("split")  # stale; recomputed later anyway
                logger.info(f"MD5 pass: reused cached inventory ({len(records)} records).")
                return
        except (json.JSONDecodeError, KeyError):
            pass

    n_failed = 0
    for r in tqdm(records, desc="MD5 pass (decode + hash)"):
        md5 = hash_image(Path(r["path"]))
        if md5 is None:
            r["split"] = SPLIT_FAIL_DECODE
            n_failed += 1
        else:
            r["md5"] = md5
    inv_path.write_text(json.dumps({
        "keys": expected,
        "records": [{"dataset_source": r["dataset_source"], "name": r["name"],
                     "md5": r["md5"], "split": r["split"]} for r in records]},
        indent=1))
    logger.info(f"MD5 pass done: {len(records)} images, {n_failed} decode failures.")


# ---------------------------------------------------------------------------
# Phase 2: cached AntelopeV2 embeddings (resumable, memmap .npy)
# ---------------------------------------------------------------------------
def embedding_cache(embed_records: List[dict], cache_dir: Path) -> np.ndarray:
    """(n_embed, 512) float32 memmap; zero rows = not-yet-embedded or failed.
    Cache is recreated whenever the record list changes."""
    npy_path = cache_dir / "embeddings.npy"
    meta_path = cache_dir / "embed_meta.json"
    expected_keys = [_record_key(r) for r in embed_records]

    reuse = False
    if npy_path.exists() and meta_path.exists():
        try:
            reuse = json.loads(meta_path.read_text("utf-8")).get("keys") == expected_keys
        except (json.JSONDecodeError, KeyError):
            pass
    if reuse:
        logger.info(f"Reusing cached embeddings ({npy_path.name}, {len(embed_records)} rows).")
        return np.lib.format.open_memmap(str(npy_path), mode="r+")

    logger.info(f"Creating fresh embedding cache ({len(embed_records)} rows) ...")
    arr = np.lib.format.open_memmap(str(npy_path), mode="w+", dtype="float32",
                                    shape=(len(embed_records), EMBEDDING_DIM))
    arr[:] = 0.0
    arr.flush()
    meta_path.write_text(json.dumps({"keys": expected_keys}, indent=1))
    return arr


def run_embedding_pass(embed_records: List[dict], arr: np.ndarray,
                       aligned: bool) -> int:
    """Embed every record without a nonzero cached row. Returns attempted
    count; failures (exception OR all-zero output) are marked on the record."""
    attempted = 0
    for i, r in enumerate(tqdm(embed_records, desc="Embedding (AntelopeV2)")):
        if np.linalg.norm(arr[i]) > 1e-9:
            continue                                     # already cached
        attempted += 1
        try:
            res = embed_image(r["path"], is_aligned=aligned)
            emb = np.asarray(res.embedding, dtype=np.float32).reshape(-1)
            if emb.shape[0] != EMBEDDING_DIM:
                raise RuntimeError(f"unexpected embedding dim {emb.shape[0]}")
            arr[i] = emb
            arr.flush()
            if np.linalg.norm(arr[i]) <= 1e-9:           # model returned all zeros
                raise RuntimeError("embedding norm is ~0 (blank/degenerate crop)")
        except Exception as e:                            # noqa: BLE001
            logger.warning(f"embedding failed for {r['path']}: {e}")
            r["split"] = SPLIT_FAIL_EMBED
            arr[i] = 0.0
            arr.flush()
    return attempted


# ---------------------------------------------------------------------------
# Phase 3: identity clustering -- chunked cosine edges + union-find
# ---------------------------------------------------------------------------
def collect_edges(X: np.ndarray, floor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All pairs with cosine >= floor as (rows_i, cols_j, sim_float16), upper
    triangle only. Fully vectorized per block."""
    n = X.shape[0]
    XT = X.astype(np.float32, copy=False).T
    rows_acc: List[np.ndarray] = []
    cols_acc: List[np.ndarray] = []
    sim_acc: List[np.ndarray] = []
    for i0 in range(0, n, EMBED_BLOCK):
        hi = min(i0 + EMBED_BLOCK, n)
        block = X[i0:hi] @ XT                                  # (B, n) float32
        row_abs = i0 + np.arange(hi - i0)
        keep = (block >= floor) & (np.arange(n)[None, :] > row_abs[:, None])
        rr, cc = np.nonzero(keep)
        if rr.size:
            rows_acc.append((rr + i0).astype(np.int64))
            cols_acc.append(cc.astype(np.int64))
            sim_acc.append(block[rr, cc].astype(np.float16))
    if not rows_acc:
        return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float16))
    return (np.concatenate(rows_acc), np.concatenate(cols_acc), np.concatenate(sim_acc))


def union_find(n: int, edges: Tuple[np.ndarray, np.ndarray, np.ndarray],
               threshold: float) -> np.ndarray:
    """Connected components at cosine >= threshold; labels relabeled by each
    component's smallest member so identical input -> identical labels."""
    rows_i, cols_j, sim = edges
    mask = sim >= threshold
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(rows_i[mask].tolist(), cols_j[mask].tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    min_member: Dict[int, int] = {}
    for x in range(n):
        r = find(x)
        if r not in min_member or x < min_member[r]:
            min_member[r] = x
    new_id = {r: i for i, (r, _) in enumerate(sorted(min_member.items(), key=lambda kv: kv[1]))}
    return np.array([new_id[find(x)] for x in range(n)], dtype=np.int64)


# ---------------------------------------------------------------------------
# Phase 4: per-cluster train/test assignment
# ---------------------------------------------------------------------------
def _dest_for(r: dict, split: str) -> str:
    return f"{split}/{r['dataset_source']}_{r['name']}"


def assign_splits(records: List[dict], arr: np.ndarray, threshold: float,
                  rep_rule: str, singleton_train: bool,
                  seed: int) -> Tuple[int, int, Tuple]:
    """
    Core of the merge. Sets split/cluster_id/dest_name on every embedded
    record. Returns (n_clusters, n_edges, edges) for the report.

    Order of operations (the two duplicate notions never fight):
      1. Cluster by identity over ALL embedded rows (cosine >= threshold).
      2. Per cluster: choose the single test representative (rule below).
      3. Collapse exact-photo copies: for each distinct MD5 in the cluster,
         at most one copy may be kept in train (prefer the highest-priority
         dataset); every other byte-identical copy is dropped_exact_dup.
    """
    valid = np.linalg.norm(arr, axis=1) > 1e-9
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0, 0, (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float16))

    X = np.asarray(arr[valid], dtype=np.float32)
    edges = collect_edges(X, min(EDGE_FLOOR, threshold))
    sim = edges[2]
    labels = union_find(n_valid, edges, threshold)

    # valid positions (into embed_records) in the same order as X rows
    valid_pos = np.nonzero(valid)[0]

    clusters: Dict[int, List[int]] = defaultdict(list)
    for j, lab in enumerate(labels.tolist()):
        clusters[int(lab)].append(int(valid_pos[j]))

    rng = random.Random(seed)

    def rec_key(r: dict) -> tuple:
        return _priority_key(r["dataset_source"]) + (r["name"],)

    used_dests: set = set()
    n_dup_dropped = 0

    for lab, pos_list in sorted(clusters.items()):
        group = [records[p] for p in pos_list]            # records order == embed order

        if len(group) == 1:
            rec = group[0]
            if singleton_train:
                rec["split"] = SPLIT_TRAIN
                rec["dest_name"] = _dest_for(rec, SPLIT_TRAIN)
            else:
                rec["split"] = SPLIT_SINGLETON
            rec["cluster_id"] = lab
            continue

        # 1 test representative per identity cluster
        if rep_rule == "priority":
            rep = min(group, key=rec_key)
        else:                                             # random (seeded, deterministic)
            rep = rng.choice(sorted(group, key=lambda r: (r["dataset_source"], r["name"])))
        rep["cluster_id"] = lab
        rep["split"] = SPLIT_TEST
        rep["dest_name"] = _dest_for(rep, SPLIT_TEST)

        # collapse exact-photo copies per distinct MD5
        by_md5: Dict[str, List[dict]] = defaultdict(list)
        for rec in group:
            by_md5[rec["md5"]].append(rec)

        for md5, recs in sorted(by_md5.items()):
            if md5 == rep["md5"]:
                for rec in recs:
                    if rec is not rep:
                        rec["cluster_id"] = lab
                        rec["split"] = SPLIT_DUP
                        n_dup_dropped += 1
                continue
            best = min(recs, key=rec_key)
            best["cluster_id"] = lab
            best["split"] = SPLIT_TRAIN
            best["dest_name"] = _dest_for(best, SPLIT_TRAIN)
            for rec in recs:
                if rec is not best:
                    rec["cluster_id"] = lab
                    rec["split"] = SPLIT_DUP
                    n_dup_dropped += 1

    # guard against any accidental dest-name collision (shouldn't happen)
    n_renamed = 0
    for r in records:
        if r["split"] not in (SPLIT_TRAIN, SPLIT_TEST):
            continue
        d = r["dest_name"]
        if d in used_dests:
            stem = Path(d).stem
            r["dest_name"] = f"{Path(d).parent}/{stem}_{n_renamed}{Path(d).suffix}"
            n_renamed += 1
            logger.warning(f"dest name collision on {d}; renamed to {r['dest_name']}")
        used_dests.add(r["dest_name"])

    if n_renamed:
        logger.warning(f"{n_renamed} destination name(s) had to be uniquified.")

    n_clusters = len(clusters)
    logger.info(f"Clustering @ {threshold}: {n_valid} embeddings -> {n_clusters} identity "
                f"clusters ({sum(1 for g in clusters.values() if len(g) > 1)} multi-image), "
                f"{n_dup_dropped} byte-identical copies dropped.")
    return n_clusters, edges[0].size, edges


# ---------------------------------------------------------------------------
# Phase 5: write outputs
# ---------------------------------------------------------------------------
def write_outputs(records: List[dict], out_root: Path) -> Dict[str, int]:
    copied = Counter()
    for r in tqdm([x for x in records if x["split"] in (SPLIT_TRAIN, SPLIT_TEST)],
                  desc="Copying to train/ + test/"):
        dest = out_root / r["dest_name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r["path"], dest)
        copied[r["split"]] += 1
    return {"train": copied[SPLIT_TRAIN], "test": copied[SPLIT_TEST]}


def write_manifest(records: List[dict], out_root: Path) -> None:
    order = {"train": 0, "test": 1, "dropped_exact_dup": 2, "dropped_singleton": 3,
             "failed_decode": 4, "failed_embed": 5}
    with open(out_root / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for r in sorted(records, key=lambda x: (x["dataset_source"], x["name"])):
            writer.writerow({k: ("" if r[k] is None else r[k]) for k in MANIFEST_FIELDS})


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
def _global_exact_dup_groups(records: List[dict]) -> Counter:
    by_md5: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        if r["md5"]:
            by_md5[r["md5"]].append(r["dataset_source"])
    return Counter(len(v) for v in by_md5.values() if len(v) > 1)


def _sensitivity_table(edges: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       n_valid: int, chosen: float) -> List[dict]:
    rows = []
    for t in sorted(set(SENSITIVITY_THRESHOLDS) | {round(chosen, 4)}):
        labels = union_find(n_valid, edges, t)
        cnt = Counter(labels.tolist())
        rows.append({"threshold": t, "n_clusters": len(cnt),
                     "n_multi": sum(1 for v in cnt.values() if v > 1),
                     "n_singleton": sum(1 for v in cnt.values() if v == 1)})
    return rows


def write_report(records: List[dict], out_root: Path, datasets: List[str],
                 threshold: float, stats: dict,
                 singleton_train: bool, rep_rule: str, limit: Optional[int]) -> None:
    per_ds: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        per_ds[r["dataset_source"]][r["split"] or "unassigned"] += 1

    L: List[str] = []
    add = L.append
    add("# merge_dedup_split report\n")
    add(f"- root datasets scanned: {len(datasets)} -> `{', '.join(datasets)}`")
    add(f"- identity clustering threshold: {threshold}; limit: {limit or 'none'}")
    add(f"- test representative rule: `{rep_rule}`; singleton identities -> "
        f"{'train' if singleton_train else 'dropped'}\n")
    add("## Per-dataset counts\n")
    add("| dataset | total | train | test | dropped_exact_dup | dropped_singleton |"
        " failed_decode | failed_embed |")
    add("|---|---|---|---|---|---|---|---|")
    for d in datasets:
        c = per_ds[d]
        add(f"| {d} | {sum(c.values())} | {c[SPLIT_TRAIN]} | {c[SPLIT_TEST]} |"
            f" {c[SPLIT_DUP]} | {c[SPLIT_SINGLETON]} | {c[SPLIT_FAIL_DECODE]} |"
            f" {c[SPLIT_FAIL_EMBED]} |")
    add("")
    add("## Overall\n")
    add(f"- images scanned: {len(records)}")
    add(f"- identity clusters: {stats['n_clusters']} "
        f"(multi-image: {stats['n_multi']}, singletons: {stats['n_singleton']})")
    add(f"- train/ copies: {stats['copied']['train']}; test/ copies: {stats['copied']['test']}")
    add(f"- byte-identical copies dropped (duplication removed): {stats['n_dup_dropped']}")
    add(f"- decode failures: {stats['failed_decode']}; embed failures: {stats['failed_embed']}\n")

    add("## Largest identity clusters\n")
    sizes = Counter()
    cds: Dict[int, set] = defaultdict(set)
    for r in records:
        if r["cluster_id"] is None:
            continue
        cid = r["cluster_id"]
        sizes[cid] += 1
        cds[cid].add(r["dataset_source"])
    add("| size | clusters | span datasets (max) |")
    add("|---|---|---|")
    for size in sorted(set(sizes.values()), reverse=True)[:20]:
        n_cl = sum(1 for v in sizes.values() if v == size)
        maxspan = max((len(cds[c]) for c, v in sizes.items() if v == size), default=0)
        add(f"| {size} | {n_cl} | {maxspan} |")

    add("\n## Exact-photo duplication (global MD5 groups)\n")
    dup_counts = _global_exact_dup_groups(records)
    if dup_counts:
        add("Same photo found in this many folders | number of such photos")
        add("---|---")
        for k in sorted(dup_counts):
            add(f"{k} | {dup_counts[k]}")
        add(f"\nTotal redundant byte-identical copies: {sum(k * v for k, v in dup_counts.items())}")
    else:
        add("None.")

    add("\n## Threshold sensitivity (re-clustering on stored edges)\n")
    n_valid = stats["n_valid"]
    if n_valid and stats["n_edges"]:
        add("| threshold | n_clusters | multi-image | singletons |")
        add("|---|---|---|---|")
        for row in _sensitivity_table(stats["edges"], n_valid, threshold):
            add(f"| {row['threshold']} | {row['n_clusters']} | {row['n_multi']} |"
                f" {row['n_singleton']} |")
        add("\nInterpretation: if lowering the threshold adds few clusters and the"
            " biggest cluster sizes grow suddenly, over-merging (transitive chains) is"
            " likely; prefer a threshold where multi-image count is stable.")
    else:
        add("No edges (nothing similar above floor).")

    add("\n## Known limitations / difficulties\n")
    add("- Byte-identical detection is decode-level MD5. A photo re-encoded to"
        " slightly different pixels (different JPEG encoder/quality) is NOT flagged"
        " exact-dup; it is only merged if the embedding clusters it with the original.")
    add("- Identity clustering is transitive: A~B and B~C at threshold merges A,C"
        " even if A,C are less similar (chaining). Check the sensitivity table above.")
    add("- Cluster size is capped by nothing; a celebrity appearing in all 6 datasets"
        " forms one big cluster by design (1 test + rest train).")
    if not singleton_train:
        add("- Singleton identities (never duplicated) were dropped from the output"
            " (--no-singleton-train).")

    (out_root / "report.md").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge face datasets under --root, dedup by identity, "
                    "split 1-image-to-test / rest-to-train.")
    parser.add_argument("--root", required=True,
                        help="Root folder; every top-level subfolder with images = one dataset.")
    parser.add_argument("--out", default=None,
                        help="Output root (train/, test/, manifest.csv, report.md). "
                             "Default: <root>_split. MUST NOT be inside --root.")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Cosine threshold for same-identity clustering (default 0.50).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke-test subset: round-robin slice across datasets.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --rep-rule random.")
    parser.add_argument("--rep-rule", choices=["priority", "random"], default="priority",
                        help="How the single test representative per identity is chosen "
                             "(default: highest-priority dataset, then filename).")
    parser.add_argument("--no-singleton-train", action="store_true",
                        help="Drop singleton identities (N=1) instead of keeping them in train.")
    parser.add_argument("--no-aligned", action="store_false", dest="aligned", default=True,
                        help="Images are NOT pre-aligned (VGG/PK-Face raw sources) -> run "
                             "RetinaFace detection+alignment. Default: CASIA-* crops are "
                             "pre-aligned 112x112 and skip detection.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(f"--root does not exist: {root}")
    out_root = Path(args.out) if args.out else root.parent / (root.name + "_split")
    if out_root.resolve() == root.resolve() or str(out_root.resolve()).startswith(str(root.resolve()) + "/") \
            or str(root.resolve()).startswith(str(out_root.resolve()) + "/"):
        raise ValueError(f"--out {out_root} must not be inside --root {root} "
                         "(a re-run would scan its own output as a dataset).")
    cache_dir = out_root / "_cache"
    aligned = args.aligned            # True unless --no-aligned
    singleton_train = not args.no_singleton_train

    datasets = discover_datasets(root)
    if not datasets:
        raise FileNotFoundError(f"No image-containing subfolders found under {root}.")
    logger.info(f"Datasets discovered: {datasets}")

    records = collect_images(root, datasets, args.limit)
    logger.info(f"{len(records)} image(s) selected ({args.limit or 'all'} mode).")

    run_md5_pass(records, cache_dir)
    embed_records = [r for r in records if r["md5"] is not None]

    if embed_records:
        arr = embedding_cache(embed_records, cache_dir)
        run_embedding_pass(embed_records, arr, aligned=aligned)
        n_clusters, n_edges, edges = assign_splits(
            embed_records, arr, args.threshold, args.rep_rule,
            singleton_train, args.seed)
        # A cached zero row with no failure mark (e.g. resumed from an old
        # cache) is a failed embedding -- mark it so nothing is copied.
        norms = np.linalg.norm(np.asarray(arr), axis=1)
        for r, norm in zip(embed_records, norms):
            if r["split"] is None and norm <= 1e-9:
                r["split"] = SPLIT_FAIL_EMBED
        n_valid = int((norms > 1e-9).sum())
    else:
        n_clusters = n_edges = 0
        n_valid = 0
        arr = None
        edges = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float16))
        logger.warning("No decodable images -- nothing to cluster.")

    out_root.mkdir(parents=True, exist_ok=True)
    copied = write_outputs(records, out_root)
    write_manifest(records, out_root)

    n_failed_decode = sum(1 for r in records if r["split"] == SPLIT_FAIL_DECODE)
    n_failed_embed = sum(1 for r in records if r["split"] == SPLIT_FAIL_EMBED)
    n_dup = sum(1 for r in records if r["split"] == SPLIT_DUP)

    # For report/sensitivity we need cluster counts over valid embeddings:
    cs: Counter = Counter()
    if embed_records and n_valid:
        for r, norm in zip(embed_records, norms):
            if norm > 1e-9 and r["cluster_id"] is not None:
                cs[r["cluster_id"]] += 1
    stats = {"n_clusters": n_clusters,
             "n_multi": sum(1 for v in cs.values() if v > 1),
             "n_singleton": sum(1 for v in cs.values() if v == 1),
             "n_valid": n_valid,
             "n_edges": n_edges, "edges": edges,
             "copied": copied, "n_dup_dropped": n_dup,
             "failed_decode": n_failed_decode, "failed_embed": n_failed_embed}

    write_report(records, out_root, datasets, args.threshold, stats,
                 singleton_train, args.rep_rule, args.limit)

    logger.info(f"Done. train={copied['train']}, test={copied['test']}, "
                f"exact-dup dropped={n_dup}, decode failures={n_failed_decode}, "
                f"embed failures={n_failed_embed}. "
                f"Manifest: {out_root / 'manifest.csv'}; report: {out_root / 'report.md'}.")


if __name__ == "__main__":
    main()
