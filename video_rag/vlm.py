"""Offline structured visual captioning for browser-compatible video scenes."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest_io import ROOT, project_path, read_csv, write_csv


DEFAULT_SCENES = ROOT / "data_video" / "manifests" / "scene_manifest.csv"
DEFAULT_INVENTORY = ROOT / "data_video" / "manifests" / "video_source_inventory.csv"
DEFAULT_RUNS = ROOT / "data_video" / "manifests" / "vlm_runs.csv"
DEFAULT_CAPTIONS = ROOT / "data_video" / "manifests" / "vlm_scene_captions.csv"
DEFAULT_FRAME_ROOT = ROOT / "data_video" / "vlm_frames"
DEFAULT_MODEL_CACHE = ROOT / "models" / "qwen3-vl-8b"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT_VERSION = "catalog-grounded-visible-scene-json-v2"
RUN_FIELDS = [
    "scene_id",
    "record_id",
    "status",
    "frame_count",
    "frame_paths_json",
    "model",
    "model_revision",
    "device",
    "dtype",
    "prompt_version",
    "processed_at",
    "error",
]
CAPTION_FIELDS = [
    "caption_id",
    "scene_id",
    "record_id",
    "start_seconds",
    "end_seconds",
    "summary_en",
    "summary_zh",
    "product_name",
    "catalog_product_class",
    "visible_components_json",
    "visible_actions_json",
    "visible_states_json",
    "safety_or_warning_text_json",
    "uncertainty",
    "caption_text",
    "model",
    "model_revision",
    "prompt_version",
    "frame_paths_json",
    "review_status",
]
REQUIRED_PAYLOAD = {
    "summary_en": str,
    "summary_zh": str,
    "product_name": str,
    "visible_components": list,
    "visible_actions": list,
    "visible_states": list,
    "safety_or_warning_text": list,
    "uncertainty": str,
}


def successful_scenes(path: Path) -> list[dict[str, str]]:
    """Load successful scenes and reject incomplete or duplicate identifiers."""
    rows = [row for row in read_csv(path) if row.get("status") == "success"]
    if not rows:
        raise ValueError(f"No successful scenes in {path}")
    identifiers = [row["scene_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate scene identifiers")
    return rows


def frame_offsets(duration: float, maximum: int = 3) -> list[float]:
    """Choose deterministic interior timestamps with density based on duration."""
    if duration <= 0:
        raise ValueError("Scene duration must be positive")
    count = 1 if duration < 3 else 2 if duration < 8 else maximum
    fractions = {1: (0.5,), 2: (0.2, 0.8), 3: (0.1, 0.5, 0.9)}[count]
    return [min(max(duration * fraction, 0.05), max(duration - 0.05, 0.05)) for fraction in fractions]


def extract_scene_frames(
    scene: dict[str, str], frame_root: Path = DEFAULT_FRAME_ROOT, maximum: int = 3
) -> list[Path]:
    """Extract one to three audit frames from a scene clip using bundled FFmpeg."""
    import imageio_ffmpeg

    if not frame_root.is_absolute():
        frame_root = project_path(str(frame_root))
    clip = project_path(scene["clip_path"])
    if not clip.is_file():
        raise ValueError(f"Missing scene clip: {scene['scene_id']}")
    duration = float(scene["end_seconds"]) - float(scene["start_seconds"])
    output_dir = frame_root / scene["scene_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for index, offset in enumerate(frame_offsets(duration, maximum), start=1):
        output = output_dir / f"frame_{index:02d}.jpg"
        # Some short source clips contain a long audio stream but only one
        # decodable video frame. Fall back to the first frame in that case.
        for candidate_offset in (offset, 0.0):
            output.unlink(missing_ok=True)
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{candidate_offset:.3f}",
                    "-i",
                    str(clip),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    "-y",
                    str(output),
                ],
                check=True,
            )
            if output.is_file() and output.stat().st_size > 0:
                break
        if not output.is_file() or output.stat().st_size == 0:
            raise ValueError(f"Frame extraction failed: {output}")
        outputs.append(output)
    return outputs


def caption_prompt(catalog_product_class: str) -> str:
    """Return the versioned evidence-only structured captioning instruction."""
    context = f"""
You are describing ordered frames sampled from one product video scene.
The authorized source inventory labels the product category as
{json.dumps(catalog_product_class, ensure_ascii=False)}. Use this catalog label
only to disambiguate the generic device category. Do not infer a model, feature,
specification, or action merely from the label.
""".strip()
    instruction = """
Report only facts directly visible in these frames. Do not infer hidden steps,
causes, specifications, hazards, or instructions. If an item cannot be
identified from the frames, state that in uncertainty.

Return exactly one JSON object and no Markdown with these keys:
{
  "summary_en": "concise English description of visible scene",
  "summary_zh": "与英文等义的简洁中文描述",
  "product_name": "generic visible product name or unknown",
  "visible_components": ["visible component or control"],
  "visible_actions": ["visible human or machine action"],
  "visible_states": ["visible state or change"],
  "safety_or_warning_text": ["warning or safety text visibly readable in frame"],
  "uncertainty": "what is ambiguous, occluded, or not verifiable"
}
Use empty arrays when nothing is visibly supported. Never invent warning text.
""".strip()
    return f"{context}\n\n{instruction}"


def parse_caption_payload(text: str) -> dict[str, Any]:
    """Extract and strictly validate one JSON object from model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model output contains no JSON object")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Caption payload is not an object")
    uncertainty = payload.get("uncertainty")
    if uncertainty is None:
        payload["uncertainty"] = ""
    elif isinstance(uncertainty, list) and all(
        isinstance(item, str) for item in uncertainty
    ):
        payload["uncertainty"] = "; ".join(
            item.strip() for item in uncertainty if item.strip()
        )
    for key, expected_type in REQUIRED_PAYLOAD.items():
        value = payload.get(key)
        if not isinstance(value, expected_type):
            raise ValueError(f"Caption field {key!r} must be {expected_type.__name__}")
        if expected_type is list and any(not isinstance(item, str) for item in value):
            raise ValueError(f"Caption field {key!r} must contain only strings")
    if not payload["summary_en"].strip() or not payload["summary_zh"].strip():
        raise ValueError("Bilingual summaries must not be empty")
    return payload


def render_caption_text(payload: dict[str, Any]) -> str:
    """Build bilingual retrieval text while retaining structured source fields."""
    pieces = [
        f"Visual summary: {payload['summary_en'].strip()}",
        f"视觉摘要: {payload['summary_zh'].strip()}",
    ]
    labels = (
        ("Visible product", "product_name"),
        ("Visible components", "visible_components"),
        ("Visible actions", "visible_actions"),
        ("Visible states", "visible_states"),
        ("Visible warnings", "safety_or_warning_text"),
        ("Visual uncertainty", "uncertainty"),
    )
    for label, key in labels:
        value = payload[key]
        if isinstance(value, list):
            value = "; ".join(item.strip() for item in value if item.strip())
        else:
            value = value.strip()
        if value:
            pieces.append(f"{label}: {value}")
    return "\n".join(pieces)


def generate_text(model: Any, processor: Any, frame_paths: list[Path], prompt: str) -> str:
    """Run deterministic multimodal generation for one ordered frame set."""
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": str(path)} for path in frame_paths],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated = model.generate(**inputs, max_new_tokens=600, do_sample=False)
    trimmed = [
        output[len(source) :] for source, output in zip(inputs.input_ids, generated)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def caption_scene(
    model: Any,
    processor: Any,
    scene: dict[str, str],
    model_name: str,
    model_revision: str,
    device: str,
    dtype: str,
    frame_root: Path,
    catalog_product_class: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Extract frames, generate structured text, and create reproducible rows."""
    frames = extract_scene_frames(scene, frame_root)
    raw = generate_text(
        model, processor, frames, caption_prompt(catalog_product_class)
    )
    try:
        payload = parse_caption_payload(raw)
    except (ValueError, json.JSONDecodeError):
        repair = (
            caption_prompt(catalog_product_class)
            + "\nYour previous response was invalid. Return the corrected JSON object only."
        )
        payload = parse_caption_payload(generate_text(model, processor, frames, repair))
    relative_frames = [str(path.relative_to(ROOT)).replace("\\", "/") for path in frames]
    frame_json = json.dumps(relative_frames, ensure_ascii=False, separators=(",", ":"))
    now = datetime.now(timezone.utc).isoformat()
    run = {
        "scene_id": scene["scene_id"],
        "record_id": scene["record_id"],
        "status": "success",
        "frame_count": str(len(frames)),
        "frame_paths_json": frame_json,
        "model": model_name,
        "model_revision": model_revision,
        "device": device,
        "dtype": dtype,
        "prompt_version": PROMPT_VERSION,
        "processed_at": now,
        "error": "",
    }
    caption = {
        "caption_id": f"{scene['scene_id']}-vlm-0001",
        "scene_id": scene["scene_id"],
        "record_id": scene["record_id"],
        "start_seconds": scene["start_seconds"],
        "end_seconds": scene["end_seconds"],
        "summary_en": payload["summary_en"].strip(),
        "summary_zh": payload["summary_zh"].strip(),
        "product_name": payload["product_name"].strip(),
        "catalog_product_class": catalog_product_class,
        "visible_components_json": json.dumps(
            payload["visible_components"], ensure_ascii=False, separators=(",", ":")
        ),
        "visible_actions_json": json.dumps(
            payload["visible_actions"], ensure_ascii=False, separators=(",", ":")
        ),
        "visible_states_json": json.dumps(
            payload["visible_states"], ensure_ascii=False, separators=(",", ":")
        ),
        "safety_or_warning_text_json": json.dumps(
            payload["safety_or_warning_text"],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "uncertainty": payload["uncertainty"].strip(),
        "caption_text": render_caption_text(payload),
        "model": model_name,
        "model_revision": model_revision,
        "prompt_version": PROMPT_VERSION,
        "frame_paths_json": frame_json,
        "review_status": "pending",
    }
    return run, caption


def caption_collection(
    scenes_path: Path = DEFAULT_SCENES,
    runs_path: Path = DEFAULT_RUNS,
    captions_path: Path = DEFAULT_CAPTIONS,
    frame_root: Path = DEFAULT_FRAME_ROOT,
    model_name: str = DEFAULT_MODEL,
    model_cache: Path = DEFAULT_MODEL_CACHE,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    offline: bool = False,
    scene_ids: set[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load one VLM and caption every accepted scene with incremental manifests."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype_value = getattr(torch, dtype)
    model_cache.mkdir(parents=True, exist_ok=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        cache_dir=model_cache,
        local_files_only=offline,
        dtype=dtype_value,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=model_cache,
        local_files_only=offline,
    )
    revision = str(getattr(model.config, "_commit_hash", "") or "unresolved")
    runs: list[dict[str, str]] = []
    captions: list[dict[str, str]] = []
    scenes = successful_scenes(scenes_path)
    inventory = {row["record_id"]: row for row in read_csv(DEFAULT_INVENTORY)}
    missing_inventory = sorted(
        {scene["record_id"] for scene in scenes} - set(inventory)
    )
    if missing_inventory:
        raise ValueError(f"Scenes lack source inventory rows: {missing_inventory}")
    if scene_ids:
        scenes = [scene for scene in scenes if scene["scene_id"] in scene_ids]
        found = {scene["scene_id"] for scene in scenes}
        if found != scene_ids:
            raise ValueError(f"Unknown requested scene IDs: {sorted(scene_ids - found)}")
    for scene in scenes:
        print(f"vlm_caption={scene['scene_id']}", flush=True)
        try:
            run, caption = caption_scene(
                model,
                processor,
                scene,
                model_name,
                revision,
                device,
                dtype,
                frame_root,
                inventory[scene["record_id"]]["product_class"],
            )
            captions.append(caption)
        except Exception as exc:
            run = {field: "" for field in RUN_FIELDS}
            run.update(
                {
                    "scene_id": scene["scene_id"],
                    "record_id": scene["record_id"],
                    "status": "failed",
                    "model": model_name,
                    "model_revision": revision,
                    "device": device,
                    "dtype": dtype,
                    "prompt_version": PROMPT_VERSION,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        runs.append(run)
        write_csv(runs_path, RUN_FIELDS, runs)
        write_csv(captions_path, CAPTION_FIELDS, captions)
    return runs, captions
