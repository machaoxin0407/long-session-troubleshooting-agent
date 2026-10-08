"""Pinned loader for the official Qwen3-VL-Embedding implementation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .manifest_io import ROOT


DEFAULT_VISUAL_MODEL = ROOT / "models" / "Qwen3-VL-Embedding-2B"
DEFAULT_VISUAL_IMPLEMENTATION = DEFAULT_VISUAL_MODEL / "scripts"
VISUAL_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
VISUAL_MODEL_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
VISUAL_IMPLEMENTATION_REVISION = "36d45865735be96a1278a21c132ff640e2ae68ca"
VISUAL_QUERY_INSTRUCTION = (
    "Retrieve the product-support video scene that best shows the object, "
    "component, state, action, or operating condition described by the query."
)


def load_visual_embedder(
    model_path: Path = DEFAULT_VISUAL_MODEL,
    implementation_root: Path = DEFAULT_VISUAL_IMPLEMENTATION,
    *,
    fps: float = 1.0,
    max_frames: int = 8,
    total_pixels: int = 4_194_304,
) -> Any:
    """Load the pinned official model in BF16 using PyTorch SDPA."""
    import torch

    resolved_implementation = implementation_root.resolve()
    if not (resolved_implementation / "qwen3_vl_embedding.py").is_file():
        raise FileNotFoundError(
            f"Qwen visual embedding implementation is missing: "
            f"{resolved_implementation}"
        )
    sys.path.insert(0, str(resolved_implementation))
    from qwen3_vl_embedding import Qwen3VLEmbedder

    return Qwen3VLEmbedder(
        model_name_or_path=str(model_path.resolve()),
        max_length=8192,
        total_pixels=total_pixels,
        fps=fps,
        max_frames=max_frames,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
