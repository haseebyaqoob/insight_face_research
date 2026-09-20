"""
app.py
======
Stage B backend. Splits the online flow into the two operations the UI offers:

  * Ingest  : POST /ingest  -- upload image(s) -> embed -> INSERT into Milvus.
              (No search. Makes the uploaded faces immediately searchable.)
  * Search  : POST /search  -- upload a query image -> embed -> top-K hits.
              (Does NOT insert the query into Milvus.)

Extra endpoints:
  GET /db-status            -- collection health + row count + resolved index engine
  GET /file?path=...        -- serve a stored image so top-K results render as <img>
  POST /upload              -- backwards-compatible alias of /search

The uploaded query / ingest images are saved to disk under uploads/ and the
backend also serves the simple frontend/ folder as static files.

Run with:
    uvicorn app:app --reload --port 8000
Then open http://localhost:8000 in a browser.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from embedder import (
    get_embedder,
    NoFaceDetectedError,
    MODEL_VERSION,
    EMBEDDING_VERSION,
    ALIGNMENT_VERSION,
)
from milvus_client import VectorDBClient, DEFAULT_URI, DEFAULT_COLLECTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("app")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

DATA_DIR = os.environ.get("DATA_DIR", "")

app = FastAPI(title="Face Similarity Search")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_db = VectorDBClient(uri=DEFAULT_URI, collection_name=DEFAULT_COLLECTION)


@app.on_event("startup")
def startup():
    logger.info("Loading embedder (RetinaFace + AntelopeV2) -- this can take a while on first run ...")
    get_embedder()  # load once at process startup, not per-request
    _db.connect()
    _db.load_collection()
    logger.info("Backend ready.")
    try:
        logger.info(f"Collection index in use: {_db.describe_index()}")
    except Exception as e:  # noqa: BLE001 -- startup must not die on a reporting call
        logger.warning(f"Could not read index info at startup: {e}")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _save_upload(file: UploadFile) -> Path:
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


def _image_url(path: str) -> str:
    """Frontend renders hits with <img src=image_url>; the backend builds the
    URL so client-side path/backslash mangling never happens."""
    return f"/file?path={quote(path)}"


def _resolve_stored_path(raw: str) -> Optional[Path]:
    if not DATA_DIR:
        fp = Path(raw.replace("\\", "/"))
        return fp if fp.is_file() else None
    filename = Path(raw.replace("\\", "/")).name
    fp = Path(DATA_DIR) / filename
    return fp if fp.is_file() else None


def _run_search(file: UploadFile, top_k: int) -> dict:
    """Upload a query image -> embed -> top-K cosine hits. Query image is NOT
    inserted into Milvus -- searching and storing stay separate."""
    dest = _save_upload(file)

    embedder = get_embedder()
    try:
        result = embedder.embed_image(str(dest))
    except NoFaceDetectedError:
        raise HTTPException(status_code=422, detail="No face detected in the uploaded image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    try:
        hits = _db.search(result.embedding, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector search failed: {e}")

    return {
        "uploaded_image": str(dest),
        "uploaded_image_url": _image_url(str(dest)),
        "num_faces_detected": result.num_faces_detected,
        "top_k": top_k,
        "results": [
            {
                "image_path": h.image_path,
                "image_url": _image_url(h.image_path),
                "dataset_source": h.dataset_source,
                "person_id": h.person_id,
                "score": h.score,
            }
            for h in hits
        ],
    }


# ---------------------------------------------------------------------------
# Search endpoints
# ---------------------------------------------------------------------------

@app.post("/search")
async def search(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=50)):
    return _run_search(file, top_k)


@app.post("/upload")
async def upload_and_search(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=50)):
    """Backwards-compatible alias of /search kept for any older clients/scripts."""
    return _run_search(file, top_k)


# ---------------------------------------------------------------------------
# Ingest endpoint: embed + store (no search)
# ---------------------------------------------------------------------------

@app.post("/ingest")
async def ingest_images(files: List[UploadFile] = File(...)):
    """Upload one or more images, embed each, and INSERT into the collection
    so subsequent similarity searches can find them. Returns per-file
    outcomes plus the updated row count for a UI confirmation."""
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")

    embedder = get_embedder()
    out = []
    indexed = 0

    for f in files:
        filename = f.filename or "unnamed"
        dest = _save_upload(f)
        try:
            result = embedder.embed_image(str(dest))
        except NoFaceDetectedError:
            out.append({"filename": filename, "status": "skipped", "reason": "no face detected"})
            continue
        except Exception as e:
            out.append({"filename": filename, "status": "error", "reason": str(e)})
            continue

        try:
            _db.insert_embedding(
                embedding=result.embedding,
                image_path=str(dest),
                dataset_source="upload",
                person_id="",
                model_version=MODEL_VERSION,
                embedding_version=EMBEDDING_VERSION,
                alignment_version=ALIGNMENT_VERSION,
            )
        except Exception as e:
            out.append({"filename": filename, "status": "error", "reason": f"insert failed: {e}"})
            continue

        indexed += 1
        out.append({
            "filename": filename,
            "status": "indexed",
            "num_faces_detected": result.num_faces_detected,
            "image_url": _image_url(str(dest)),
        })

    try:
        row_count = _db.count()
    except Exception as e:  # noqa: BLE001
        row_count = None
        logger.warning(f"Could not read row count after ingest: {e}")

    return {"indexed": indexed, "total": len(files), "results": out, "db_row_count": row_count}


# ---------------------------------------------------------------------------
# Supporting endpoints
# ---------------------------------------------------------------------------

@app.get("/file")
def serve_file(path: str = Query(...)):
    fp = _resolve_stored_path(path)
    if fp is None:
        raise HTTPException(status_code=404, detail="Image not found on this host.")
    media_type = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
    return FileResponse(fp, media_type=media_type)


@app.get("/db-status")
def db_status():
    """Powers the header line in the UI: live row count + the actual index
    engine Milvus reports (confirms HNSW after a re-bootstrap)."""
    payload = {"connected": False, "collection": _db.collection_name}
    try:
        payload["row_count"] = _db.count()
        payload.update(_db.describe_index())
        payload["connected"] = True
    except Exception as e:
        payload["error"] = str(e)
    return payload


# Serve the simple frontend last, so API routes above still take priority.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
