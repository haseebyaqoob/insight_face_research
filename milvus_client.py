"""
milvus_client.py
=================
Thin, swappable database layer. All Milvus-specific code lives here --
embedder.py never touches this module and never inserts into Milvus
directly.

By default this connects to MILVUS LITE: a local, file-backed vector DB
that ships inside pymilvus (>=2.4) and requires NO Docker / no server
process at all. This satisfies "Python/Milvus architecture first, no
Docker yet" while still giving you a real, working ANN vector search
today.

To point at a real Milvus server later (Docker or otherwise), change
DEFAULT_URI to e.g. "http://localhost:19530" -- nothing else in this file,
or in any file that imports it, needs to change.

To swap to Qdrant later: reimplement the methods on VectorDBClient against
the Qdrant client. bootstrap_pipeline.py and app.py only ever call
connect() / create_collection() / insert_embedding(s)() / search() /
delete() / load_collection() -- they never touch pymilvus directly.

Metric choice
-------------
embedder.py L2-normalizes every embedding before returning it. We use
metric_type="COSINE" explicitly (rather than assuming raw inner product on
pre-normalized vectors) so Milvus always computes a true cosine similarity
regardless of any tiny floating-point drift from exact unit norm. Higher
score = more similar.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
from pymilvus import MilvusClient as _PymilvusClient, DataType

logger = logging.getLogger("milvus_client")

DEFAULT_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
DEFAULT_COLLECTION = os.environ.get("COLLECTION_NAME", "face_embeddings_merged")

# Explicit ANN index used for every collection this module creates.
# AUTOINDEX is intentionally NOT used: it leaves the real engine to the
# server, so search is not guaranteed to run on HNSW. HNSW + COSINE is the
# deterministic choice for L2-normalized ArcFace embeddings.
INDEX_TYPE = "HNSW"
INDEX_PARAMS = {"M": 16, "efConstruction": 200}   # valid: M in [4,64], efConstruction in [8,512]


@dataclass
class SearchResult:
    id: int
    image_path: str
    dataset_source: str
    person_id: Optional[str]
    score: float           # cosine similarity, higher = more similar


class VectorDBClient:
    """Database layer used by bootstrap_pipeline.py and app.py. Neither of
    those files should import pymilvus directly -- only this module does.
    """

    def __init__(self, uri: str = DEFAULT_URI, collection_name: str = DEFAULT_COLLECTION):
        self.uri = uri
        self.collection_name = collection_name
        self._client: Optional[_PymilvusClient] = None

    def connect(self):
        logger.info(f"Connecting to Milvus at uri={self.uri}")
        self._client = _PymilvusClient(uri=self.uri)
        return self._client

    @property
    def client(self) -> _PymilvusClient:
        if self._client is None:
            self.connect()
        return self._client

    def create_collection(self, dim: int, drop_existing: bool = False):
        """dim should come from embedder.embedding_dim (determined at
        runtime from the actual loaded model), never hardcoded."""
        if drop_existing and self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        if self.client.has_collection(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' already exists, reusing it.")
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="dataset_source", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="person_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="model_version", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="embedding_version", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="alignment_version", datatype=DataType.VARCHAR, max_length=64)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=INDEX_TYPE,
            metric_type="COSINE",
            params=INDEX_PARAMS,
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(
            f"Created collection '{self.collection_name}' "
            f"(dim={dim}, index={INDEX_TYPE}, metric=COSINE)."
        )

    def load_collection(self):
        self.client.load_collection(self.collection_name)

    def describe_index(self) -> Dict[str, Optional[object]]:
        """Read the current embedding index definition so callers (and the
        UI via app.py /db-status) can show the real engine actually used for
        search, e.g. HNSW vs AUTOINDEX. Never raises: returns an 'error' key
        instead when the collection or index is missing/unreachable."""
        try:
            names = self.client.list_indexes(collection_name=self.collection_name)
        except Exception as e:  # noqa: BLE001 -- report, don't crash db-status
            return {"error": f"list_indexes failed: {e}"}

        if not names:
            return {"index_name": None, "index_type": None, "metric_type": None, "params": {}}

        target = next((n for n in names if n.lower() == "embedding"), names[0])
        try:
            idx = self.client.describe_index(
                collection_name=self.collection_name, index_name=target
            )
        except Exception as e:  # noqa: BLE001
            return {"index_name": target, "error": f"describe_index failed: {e}"}

        def g(key: str, default=None):
            if isinstance(idx, dict):
                return idx.get(key, default)
            return getattr(idx, key, default) if idx is not None else default

        params = g("params") or {}
        if not isinstance(params, dict):
            if isinstance(params, list):
                merged: Dict[str, str] = {}
                for item in params:
                    if isinstance(item, dict) and "key" in item:
                        merged[item["key"]] = item.get("value")
                    elif isinstance(item, dict) and "name" in item:
                        merged[item["name"]] = item.get("data")
                params = merged
            else:
                params = {}

        return {
            "index_name": target,
            "field_name": g("field_name"),
            "index_type": g("index_type") or g("indexType") or g("type"),
            "metric_type": g("metric_type") or g("metricType") or g("metric"),
            "params": params,
        }

    def insert_embedding(
        self,
        embedding: np.ndarray,
        image_path: str,
        dataset_source: str = "",
        person_id: str = "",
        model_version: str = "",
        embedding_version: str = "",
        alignment_version: str = "",
    ):
        return self.insert_embeddings([{
            "embedding": embedding, "image_path": image_path,
            "dataset_source": dataset_source, "person_id": person_id,
            "model_version": model_version, "embedding_version": embedding_version,
            "alignment_version": alignment_version,
        }])

    def insert_embeddings(self, rows: List[Dict]):
        data = [{
            "embedding": np.asarray(r["embedding"], dtype=np.float32).tolist(),
            "image_path": r.get("image_path", ""),
            "dataset_source": r.get("dataset_source", ""),
            "person_id": r.get("person_id") or "",
            "model_version": r.get("model_version", ""),
            "embedding_version": r.get("embedding_version", ""),
            "alignment_version": r.get("alignment_version", ""),
        } for r in rows]
        return self.client.insert(collection_name=self.collection_name, data=data)

    def image_already_indexed(self, image_path: str) -> bool:
        """Used by bootstrap_pipeline.py to skip already-processed images
        on re-runs."""
        # Escape backslashes BEFORE quotes (Windows paths are
        # backslash-separated, and the filter string below is a string
        # literal in Milvus's expression grammar).
        safe_path = image_path.replace("\\", "\\\\").replace('"', '\\"')
        res = self.client.query(
            collection_name=self.collection_name,
            filter=f'image_path == "{safe_path}"',
            output_fields=["id"],
            limit=1,
        )
        return len(res) > 0

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        results = self.client.search(
            collection_name=self.collection_name,
            data=[np.asarray(embedding, dtype=np.float32).tolist()],
            limit=top_k,
            output_fields=["image_path", "dataset_source", "person_id"],
        )
        out = []
        for hit in results[0]:
            entity = hit.get("entity", hit)
            out.append(SearchResult(
                id=hit["id"],
                image_path=entity.get("image_path", ""),
                dataset_source=entity.get("dataset_source", ""),
                person_id=entity.get("person_id") or None,
                score=float(hit["distance"]),  # COSINE metric -> this IS cosine similarity
            ))
        return out

    def delete(self, ids: List[int]):
        return self.client.delete(collection_name=self.collection_name, ids=ids)

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))
