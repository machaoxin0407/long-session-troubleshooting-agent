"""Dense scene-index persistence and local embedding-service client."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from .manifest_io import ROOT


DEFAULT_DENSE_INDEX = (
    ROOT / "data_video" / "indexes" / "video_dense_qwen3_embedding_0_6b.npz"
)
DEFAULT_VISUAL_DENSE_INDEX = (
    ROOT / "data_video" / "indexes" / "video_visual_qwen3_vl_embedding_2b.npz"
)
DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QUERY_TASK = (
    "Given a product-support question, retrieve the video scene that best shows "
    "the relevant object, component, state, or operation."
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_rows(vectors: np.ndarray) -> np.ndarray:
    """Return float32 L2-normalized rows and reject zero vectors."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Dense embeddings contain a zero vector")
    return array / norms


@dataclass(frozen=True)
class DenseSceneIndex:
    """An ordered matrix aligned one-to-one with video-scene evidence rows."""

    scene_ids: list[str]
    vectors: np.ndarray
    metadata: dict[str, object]

    def save(self, path: Path) -> None:
        """Write a compressed, pickle-free NumPy index."""
        if len(self.scene_ids) != len(self.vectors):
            raise ValueError("Scene ID and vector counts differ")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            scene_ids=np.asarray(self.scene_ids, dtype=np.str_),
            vectors=normalized_rows(self.vectors),
            metadata_json=np.asarray(
                json.dumps(self.metadata, ensure_ascii=False, sort_keys=True),
                dtype=np.str_,
            ),
        )

    @classmethod
    def load(cls, path: Path) -> "DenseSceneIndex":
        """Load and validate a pickle-free scene index."""
        with np.load(path, allow_pickle=False) as archive:
            scene_ids = [str(item) for item in archive["scene_ids"].tolist()]
            vectors = normalized_rows(archive["vectors"])
            metadata = json.loads(str(archive["metadata_json"].item()))
        if len(scene_ids) != len(vectors):
            raise ValueError(f"Malformed dense scene index: {path}")
        return cls(scene_ids=scene_ids, vectors=vectors, metadata=metadata)

    def similarities(self, query_vector: np.ndarray) -> np.ndarray:
        """Return cosine similarities in scene order."""
        query = normalized_rows(query_vector)
        if query.shape != (1, self.vectors.shape[1]):
            raise ValueError(
                f"Query embedding shape {query.shape} does not match "
                f"index dimension {self.vectors.shape[1]}"
            )
        return self.vectors @ query[0]


def request_query_embedding(
    endpoint: str,
    query: str,
    timeout_seconds: float | None = None,
    timeout_env: str = "VIDEO_DENSE_TIMEOUT_S",
) -> np.ndarray:
    """Request one normalized query vector from the isolated local service."""
    if timeout_seconds is None:
        timeout_seconds = float(os.getenv(timeout_env, "1.5"))
    url = endpoint.rstrip("/") + "/v1/embeddings/query"
    response = requests.post(
        url,
        json={"texts": [query]},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    vectors = payload.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != 1:
        raise ValueError("Embedding service returned an invalid response")
    return normalized_rows(np.asarray(vectors[0], dtype=np.float32))[0]
