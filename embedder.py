"""
embedder.py
============
Single, shared face-embedding function used by BOTH the offline bootstrap
pipeline (Stage A) and the online frontend/backend upload pipeline (Stage B).

Pipeline (frozen/versioned -- do not change without re-embedding everything
already stored in Milvus):

    image (BGR np.ndarray or path)
        -> RetinaFace face detection                (uniface)
        -> 5-point landmark -> similarity warp to ARCFACE_DST   (112x112)
        -> AntelopeV2 (glintr100.onnx)               (insightface model_zoo)
        -> L2-normalized embedding

This is the exact validated alignment path from the research/evaluation
pipeline (RetinaFace -> ARCFACE_DST warp -> AntelopeV2), NOT the CFP-FP
-specific derived-template alignment -- that was a benchmark-only
workaround for a dataset of already-cropped images and has no role in a
pipeline that has to handle arbitrary raw uploads.

This module intentionally does NOT:
    - compute accuracy / ROC / FAR-FRR
    - build verification pairs
    - decide whether two faces match
    - talk to Milvus (see milvus_client.py for that)

It is the *only* place alignment + model choice are defined. Both
bootstrap_pipeline.py and app.py import from here so a stored embedding
and a query embedding are always produced the exact same way.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional, Union, List

import cv2
import numpy as np
import requests
from skimage import transform as sktransform

logger = logging.getLogger("embedder")

# ---------------------------------------------------------------------------
# VERSIONING -- store these alongside every embedding in Milvus metadata.
# If ANY of these change (different model, different alignment target,
# different detector), existing stored embeddings are no longer directly
# comparable to newly generated ones, and the dataset must be re-embedded
# from scratch. Treat this whole module as frozen once you start populating
# Milvus for real.
# ---------------------------------------------------------------------------
MODEL_VERSION = "AntelopeV2"
EMBEDDING_VERSION = "v1"
ALIGNMENT_VERSION = "retinaface_arcface_v1"

CROP_SIZE = 112

# Validated alignment target -- the ArcFace destination landmark template.
ARCFACE_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]
], dtype=np.float32)

ANTELOPE_DIR = os.path.expanduser("~/.insightface/models/antelopev2")
ANTELOPE_REC_FILENAME = "glintr100.onnx"
# Each entry is a full URL template resolving to ANTELOPE_REC_FILENAME's
# bytes on that mirror -- NOT necessarily the same path shape, since
# different mirrors lay the repo out differently (checked live: immich-app
# nests the recognition model at recognition/model.onnx, not a flat
# glintr100.onnx at repo root; rupeshs keeps the original flat layout).
ANTELOPE_MIRRORS = [
    "https://huggingface.co/immich-app/antelopev2/resolve/main/recognition/model.onnx",
    "https://huggingface.co/rupeshs/antelopev2/resolve/main/{fname}",
]

DET_THRESH_DEFAULT = 0.5
DET_SIZE_DEFAULT = 640


class NoFaceDetectedError(Exception):
    """Raised when RetinaFace finds no face in the input image."""


@dataclass
class EmbeddingResult:
    embedding: np.ndarray                         # (D,) float32, L2-normalized
    bbox: np.ndarray                             # (4,) box of the face actually used
    num_faces_detected: int
    aligned_crop: Optional[np.ndarray] = None    # 112x112x3 BGR, only if requested
    model_version: str = MODEL_VERSION
    embedding_version: str = EMBEDDING_VERSION
    alignment_version: str = ALIGNMENT_VERSION


DOWNLOAD_MAX_RETRIES = 5          # retries per mirror before moving to the next mirror
DOWNLOAD_RETRY_BACKOFF = 3        # seconds, multiplied by attempt number
DOWNLOAD_CHUNK_SIZE = 1 << 20     # 1 MB chunks
DOWNLOAD_TIMEOUT = (30, 120)      # (connect timeout, read timeout) seconds


def _download_one_mirror(url: str, dest: str) -> bool:
    """
    Download url -> dest with resume support and Content-Length verification.
    Writes to a `dest + '.part'` file so a truncated download is never mistaken
    for a complete one. Returns True on a verified-complete download.
    """
    part = dest + ".part"

    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            resume_from = os.path.getsize(part) if os.path.exists(part) else 0
            headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

            with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, headers=headers) as r:
                if r.status_code not in (200, 206):
                    logger.warning(f"{url} returned status {r.status_code}, skipping mirror.")
                    return False

                # If the server ignored our Range request (200 instead of 206),
                # we can't safely append -- start this attempt over from scratch.
                if resume_from and r.status_code == 200:
                    resume_from = 0

                total_expected = None
                if "Content-Range" in r.headers:
                    # format: "bytes start-end/total"
                    try:
                        total_expected = int(r.headers["Content-Range"].split("/")[-1])
                    except (ValueError, IndexError):
                        pass
                elif "Content-Length" in r.headers:
                    try:
                        total_expected = resume_from + int(r.headers["Content-Length"])
                    except ValueError:
                        pass

                mode = "ab" if resume_from else "wb"
                with open(part, mode) as f:
                    for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

            actual_size = os.path.getsize(part)

            if total_expected is not None and actual_size < total_expected:
                logger.warning(
                    f"Attempt {attempt}/{DOWNLOAD_MAX_RETRIES} for {url} incomplete: "
                    f"{actual_size}/{total_expected} bytes. Will retry and resume."
                )
                import time
                time.sleep(DOWNLOAD_RETRY_BACKOFF * attempt)
                continue

            if actual_size <= 1_000_000:
                logger.warning(f"Downloaded file from {url} is suspiciously small ({actual_size} bytes).")
                import time
                time.sleep(DOWNLOAD_RETRY_BACKOFF * attempt)
                continue

            os.replace(part, dest)
            return True

        except Exception as e:
            logger.warning(
                f"Attempt {attempt}/{DOWNLOAD_MAX_RETRIES} for mirror {url} failed: {e}. "
                f"{'Retrying with resume...' if attempt < DOWNLOAD_MAX_RETRIES else 'Giving up on this mirror.'}"
            )
            import time
            time.sleep(DOWNLOAD_RETRY_BACKOFF * attempt)
            continue

    return False


def _download_antelope_weights() -> str:
    os.makedirs(ANTELOPE_DIR, exist_ok=True)
    dest = os.path.join(ANTELOPE_DIR, ANTELOPE_REC_FILENAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    for mirror_tmpl in ANTELOPE_MIRRORS:
        url = mirror_tmpl.format(fname=ANTELOPE_REC_FILENAME)
        logger.info(f"Downloading AntelopeV2 weights from {url} ...")
        if _download_one_mirror(url, dest):
            return dest
        logger.warning(f"Mirror {url} did not yield a complete file, trying next mirror.")
    raise RuntimeError(
        f"Could not download {ANTELOPE_REC_FILENAME} for AntelopeV2 from any mirror "
        f"(tried {len(ANTELOPE_MIRRORS)} mirror(s), {DOWNLOAD_MAX_RETRIES} attempts each with resume). "
        "Check your network / place the file manually at: " + dest
    )


class FaceEmbedder:
    """
    Loads RetinaFace + AntelopeV2 ONCE and exposes embed_image().
    Create a single instance per process (see get_embedder() below) --
    loading the models is slow and should not happen per-image.
    """

    def __init__(
        self,
        det_thresh: float = DET_THRESH_DEFAULT,
        det_size: int = DET_SIZE_DEFAULT,
        providers: Optional[List[str]] = None,
    ):
        import onnxruntime as ort
        from insightface.model_zoo import model_zoo
        from uniface.detection import RetinaFace as UniFaceRetinaFace

        if providers is None:
            available = ort.get_available_providers()
            if "DmlExecutionProvider" in available:
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
                logger.info("DmlExecutionProvider available -- running models on AMD GPU.")
            elif "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                logger.info("CUDAExecutionProvider available -- running models on NVIDIA GPU.")
            else:
                providers = ["CPUExecutionProvider"]
                logger.warning(
                    "Neither DmlExecutionProvider nor CUDAExecutionProvider available -- running on CPU. "
                    "This is slower than GPU-validated pipelines."
                )

        self.providers = providers
        self.det_thresh = det_thresh
        self.det_size = det_size

        # onnxruntime-gpu >= 1.21 can ship CUDA/cuDNN DLLs inside its nvidia_*
        # site-packages wheels. Preload them BEFORE any InferenceSession is built
        # so the CUDA EP reliably finds its DLLs (no PATH hacks needed).
        if any(p.startswith("CUDAExecutionProvider") for p in providers):
            try:
                ort.preload_dlls()
                logger.info("Preloaded onnxruntime CUDA/cuDNN DLLs for CUDAExecutionProvider.")
            except AttributeError:
                logger.info("This onnxruntime has no preload_dlls() -- assuming CUDA DLLs are on PATH.")
            except Exception as e:  # noqa: BLE001 -- never block startup over a preload issue
                logger.warning(f"onnxruntime.preload_dlls() failed (continuing anyway): {e}")

        logger.info(f"Loading RetinaFace detector (providers={providers}) ...")
        self.detector = UniFaceRetinaFace(
            confidence_threshold=det_thresh,
            input_size=(det_size, det_size),
            providers=providers,
        )

        logger.info("Loading AntelopeV2 (glintr100.onnx) ...")
        rec_path = _download_antelope_weights()
        self.rec_model = model_zoo.get_model(rec_path, providers=providers)
        
        has_gpu = any(p in providers for p in ["DmlExecutionProvider", "CUDAExecutionProvider"])
        self.rec_model.prepare(ctx_id=0 if has_gpu else -1)

        # Determine embedding dimensionality from the actual loaded model
        # rather than hardcoding a possibly-wrong constant.
        dummy = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        dummy_feat = self.rec_model.get_feat([dummy])
        self.embedding_dim = int(dummy_feat.shape[1])
        logger.info(f"AntelopeV2 ready. Embedding dimensionality = {self.embedding_dim}")

    def _detect(self, bgr_img: np.ndarray):
        faces = self.detector.detect(bgr_img)
        return faces or []

    @staticmethod
    def _select_face(faces):
        """Largest-area detected face wins when more than one is found."""
        def area(f):
            x1, y1, x2, y2 = f.bbox
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return max(faces, key=area)

    @staticmethod
    def _align(bgr_img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        tform = sktransform.SimilarityTransform()
        tform.estimate(landmarks.astype(np.float32), ARCFACE_DST)
        M = tform.params[0:2, :]
        return cv2.warpAffine(
            bgr_img, M, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR, borderValue=0.0
        )

    def embed_image(
        self,
        image: Union[str, np.ndarray],
        return_crop: bool = False,
        is_aligned: bool = False,
    ) -> EmbeddingResult:
        """
        image: a path to an image file, OR a BGR np.ndarray (e.g. from cv2.imread).

        is_aligned: set True for sources that are ALREADY a validated 112x112
        ArcFace-aligned crop (AgeDB-30/CALFW/CPLFW .bmp files, and any
        CASIA-<name> folder produced by casia_bin_extractor.py -- verification
        bins are pre-aligned by construction). When True, RetinaFace
        detection + landmark warp is skipped entirely and the image goes
        straight to the recognition model (resized to CROP_SIZE if it isn't
        already). This matches how these benchmark sets are meant to be
        consumed, avoids re-detecting a face inside an already-tight 112x112
        crop (RetinaFace can fail or mis-align there), and is a lot faster
        for large pre-aligned sets. Raw sources (LFW, CFP-FP, and any future
        arbitrary upload) must keep is_aligned=False.

        Raises NoFaceDetectedError if no face is found in a non-pre-aligned
        image -- callers decide what to do with that (skip during bootstrap,
        reject upload in the app). Never raised when is_aligned=True.

        On multiple detected faces (only relevant when is_aligned=False), uses
        the largest one and reports num_faces_detected so the caller can
        flag/handle ambiguous uploads.
        """
        if isinstance(image, str):
            bgr = cv2.imread(image)
            if bgr is None:
                raise ValueError(f"Could not read image at path: {image}")
        else:
            bgr = image

        if is_aligned:
            aligned = bgr
            if aligned.shape[0] != CROP_SIZE or aligned.shape[1] != CROP_SIZE:
                aligned = cv2.resize(aligned, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
            # No detector ran, so there's no real bbox/face-count -- report
            # the whole-frame box and a count of 1 rather than fabricating
            # detector output.
            bbox = np.array([0.0, 0.0, float(bgr.shape[1]), float(bgr.shape[0])], dtype=np.float32)
            num_faces_detected = 1
        else:
            faces = self._detect(bgr)
            if not faces:
                raise NoFaceDetectedError("No face detected in the provided image.")

            face = self._select_face(faces)
            landmarks = np.asarray(face.landmarks, dtype=np.float32)
            aligned = self._align(bgr, landmarks)
            bbox = np.asarray(face.bbox, dtype=np.float32)
            num_faces_detected = len(faces)

        feat = self.rec_model.get_feat([aligned])[0].astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 1e-9:
            feat = feat / norm

        return EmbeddingResult(
            embedding=feat,
            bbox=bbox,
            num_faces_detected=num_faces_detected,
            aligned_crop=aligned if return_crop else None,
        )


# ---------------------------------------------------------------------------
# Module-level singleton. Both bootstrap_pipeline.py and app.py should call
# get_embedder() so the (slow) model load happens once per process.
# ---------------------------------------------------------------------------
_embedder_singleton: Optional[FaceEmbedder] = None


def get_embedder(**kwargs) -> FaceEmbedder:
    global _embedder_singleton
    if _embedder_singleton is None:
        _embedder_singleton = FaceEmbedder(**kwargs)
    return _embedder_singleton


def embed_image(
    image: Union[str, np.ndarray], return_crop: bool = False, is_aligned: bool = False
) -> EmbeddingResult:
    """Functional convenience wrapper: embedding = embed_image(image)."""
    return get_embedder().embed_image(image, return_crop=return_crop, is_aligned=is_aligned)