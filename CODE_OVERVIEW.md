# Code Overview

Brief description and run instructions for each Python file in the repo root.

---

## `embedder.py`
**Purpose:** The single shared face-embedding module. Loads the ONNX model (AntelopeV2 / glintr100) and exposes `embed_image()` — takes an image (path or BGR array), runs RetinaFace detection, similarity-warps to 112x112 ArcFace alignment, and returns an L2-normalized 512-D embedding. This is the *only* place alignment + model choice are defined; both the offline bootstrap and the online backend import it so stored and query embeddings are produced identically.

**Run:** Not run directly — imported as a library by `app.py`, `ingest_merged_split.py`, `merge_dedup_split.py`, `evaluate_test_set.py`, and the tests.

---

## `app.py`
**Purpose:** Stage B backend. FastAPI server exposing `/ingest` (upload → embed → insert into Milvus), `/search` (upload query → top-K hits), `/db-status`, and `/file?path=...`. Also serves the `frontend/` folder as static files and saves uploads to `uploads/`.

**Env vars:** Reads `DATA_DIR`, `MILVUS_URI`, `COLLECTION_NAME` from `.env` file (via python-dotenv). `DATA_DIR` overrides stored image paths for cross-machine portability.

**Run:**
```
uvicorn app:app --reload --port 8000
```
Then open `http://localhost:8000` in a browser.

---

## `casia_bin_extractor.py`
**Purpose:** Parses InsightFace-format `.bin` verification files from the CASIA-WebFace Kaggle package's `eval/` folder (e.g. `lfw.bin`, `agedb_30.bin`) and extracts the embedded JPEG pairs to disk as a flat, pre-aligned folder per bin file (`CASIA-LFW/`, `CASIA-AgeDB-30/`, etc.), which `merge_dedup_split.py` auto-discovers under `--root`.

**Run:**
```
python casia_bin_extractor.py --bins-dir D:\Code\casia_kaggle\eval --out CASIA
```

---

## `dedup_with_casia.py`
**Purpose:** Standalone optional preprocessing tool — exact-pixel MD5 dedup across the existing `clean_dataset/` folders AND any new `CASIA-*` folders. Non-destructive — writes a manifest of duplicate groups and can optionally export a deduplicated copy. Only run directly: `merge_dedup_split.py` no longer imports it (its `IMAGE_EXTS` / `DEDUP_PRIORITY_FOLDERS` / `hash_image` are now inlined there).

**Run (manifest only):**
```
python dedup_with_casia.py --dataset-root D:\Code\embedding_model\clean_dataset --casia-root D:\Code\embedding_model\CASIA --manifest dup_manifest.csv
```
**Run (export deduped copy):**
```
python dedup_with_casia.py --dataset-root ... --casia-root ... --manifest ... --export-root D:\Code\embedding_model\clean_dataset_dedup
```

---

## `merge_dedup_split.py`
**Purpose:** The core Stage A step. Merges all dataset folders under a `--root` into one identity-deduplicated whole and splits it into `train/` (rest) and `test/` (one representative per identity). Identity is recovered by AntelopeV2 embedding cosine clustering (with exact-pixel MD5 as a cheap first pass). Writes `manifest.csv` and `report.md` to `--out`. Resumable via `_cache/`.

**Run (full):**
```
python merge_dedup_split.py --root "D:\Code\embedding_model\CASIA" --out "D:\Code\embedding_model\CASIA_split"
```
**Run (smoke test):**
```
python merge_dedup_split.py --root "D:\Code\embedding_model\CASIA" --out "D:\Code\embedding_model\CASIA_smoke" --limit 300 --seed 1
```

---

## `ingest_merged_split.py`
**Purpose:** Stage B. Ingests the TRAIN split produced by `merge_dedup_split.py` into a fresh Milvus collection (`face_embeddings_merged` by default) so test-set evaluation has a clean gallery. Reuses the vectors already computed by the merge step from `<split-root>/_cache` (`embeddings.npy` + `embed_meta.json`, keyed by dataset+name); only images missing from the cache are embedded fresh with the model (loaded on demand). If `_cache/` is missing it warns and embeds all.

**Run:**
```
python ingest_merged_split.py --split-root D:\Code\embedding_model\CASIA_split
python ingest_merged_split.py --split-root D:\Code\embedding_model\CASIA_split --collection-name face_embeddings_merged --drop-existing
```

---

## `evaluate_test_set.py`
**Purpose:** Stage C. Embeds every TEST-split image and searches the merged train collection. Scores hit@1 / hit@5 by checking if the top results carry the query's own `person_id` (`cluster_<cluster_id>`). Writes `results.csv` and `evaluation.md` to `--out`.

**Run:**
```
python evaluate_test_set.py --split-root D:\Code\embedding_model\CASIA_split --out eval_results
```

---

## `split_pkface.py`
**Purpose:** Split the PK-face dataset into train/test sets. Removes singletons (identities with only 1 image). Train: 1 random image per identity. Test: remaining images per identity. Outputs `manifest.csv` with full split details.

**Run:**
```
python split_pkface.py --input D:\PK-face --output D:\PK-face_split
python split_pkface.py --input /data/PK-face --output /data/PK-face_split --seed 123
```

---

## `ingest_folder.py`
**Purpose:** General-purpose script to ingest any folder of images into a Milvus collection. Creates the collection if it doesn't exist; inserts into it if it does. Supports batch insertion, skip-existing for resume, and failure logging.

**Run:**
```
python ingest_folder.py --folder D:\PK-face_split\train --collection pkface_train
python ingest_folder.py --folder /data/images --collection my_faces --batch-size 64
python ingest_folder.py --folder /data/images --collection faces --drop-existing
```

---

## `pkface_test.py`
**Purpose:** Evaluate PK-face test set against a Milvus collection. Extracts identity from PK-face filename pattern (`pa16_1949_3.jpg` → `pa16_1949`), embeds each test image, searches Milvus, and calculates Hit@1 / Hit@K accuracy. Outputs `results.csv` and `evaluation.md`.

**Run:**
```
python pkface_test.py --test-folder D:\PK-face_split\test --collection pkface_train
python pkface_test.py --test-folder /data/test --collection my_faces --top-k 10
```

---

## `.env`
**Purpose:** Configuration file for environment variables. Read by `app.py` and `milvus_client.py` via python-dotenv. Contains `DATA_DIR`, `MILVUS_URI`, `COLLECTION_NAME`.

**Contents:**
```
DATA_DIR=
MILVUS_URI=http://localhost:19530
COLLECTION_NAME=face_embeddings_merged
```

---

## `changes.md`
**Purpose:** Change log tracking all modifications to the codebase.

---

## `extract_rec_images.py`
**Purpose:** One-off extraction tool. Decodes InsightFace RecordIO training files (`train.rec` / `train.idx` / `train.lst` for CASIA-WebFace, `train.rec` / `train.idx` for VGG-Face) and writes the original JPEG bytes back to disk organized per identity: `<out>/<CASIA|VGG>/<label>/00001.jpg`. The resulting crops are pre-aligned 112x112 and can be fed into `merge_dedup_split.py` / `ingest_merged_split.py` with `is_aligned=True`.

**Run (full):**
```
python extract_rec_images.py --dataset-path D:\Code\insightface\CASIA --dataset-path D:\Code\insightface\VGG
```
**Run (smoke):**
```
python extract_rec_images.py --dataset-path D:\Code\insightface\CASIA --dataset-path D:\Code\insightface\VGG --max-images 500 --out extracted_smoke
```

---

## `milvus_client.py`
**Purpose:** Thin database layer. All pymilvus calls live here; nothing else in the project imports pymilvus directly. Defines `VectorDBClient` (connect, create_collection, insert, search, delete, count) and the `SearchResult` dataclass. Defaults to Milvus Lite at `http://localhost:19530`; swap `DEFAULT_URI` to point at a remote server.

**Env vars:** `MILVUS_URI` and `COLLECTION_NAME` can be set via `.env` file.

**Run:** Not run directly — imported as a library by `app.py`, `ingest_merged_split.py`, `evaluate_test_set.py`, `ingest_folder.py`, `pkface_test.py`, and the tests.

---

## `tests/test_embedder.py`
**Purpose:** Sanity tests for the embedder — embedding shape & L2-norm, determinism on repeat calls, and that a blank image raises `NoFaceDetectedError`.

**Run:**
```
python -m pytest tests/test_embedder.py -v
```
(or `python tests/test_embedder.py`). Note: `KNOWN_IMAGE_PATH` inside the file must point at a real face image on your machine first.

---

## `tests/test_milvus.py`
**Purpose:** End-to-end tests against a throwaway Milvus Lite collection (deleted at the end) — insertion/retrieval, exact-match-ranking, and same-person-above-different-person ordering. No absolute similarity threshold is asserted (only relative ordering).

**Run:**
```
python -m pytest tests/test_milvus.py -v
```
The `EXISTING_IMAGE_PATH`, `SAME_PERSON_OTHER_IMAGE_PATH`, and `DIFFERENT_PERSON_IMAGE_PATH` constants inside the file must point at real images on your machine first.

---

## `tests/test_ingest_eval.py`
**Purpose:** Unit tests for the pure (model/DB-free) logic of `ingest_merged_split.py` and `evaluate_test_set.py` — manifest loading/filtering, missing-file detection, no-train-identity detection, train/test overlap guard, and the accuracy-summary math.

**Run:**
```
python -m pytest tests/test_ingest_eval.py -v
```

---

## `tests/test_merge_dedup_split.py`
**Purpose:** Unit tests for the pure (model-free) logic of `merge_dedup_split.py` — edge collection / union-find clustering, split assignment, representative-priority ordering, dataset discovery, the MD5 pass, and round-robin image selection. Runs without model weights or a GPU, using hand-built embeddings and tiny real PNG files.

**Run:**
```
python -m pytest tests/test_merge_dedup_split.py -v
```
