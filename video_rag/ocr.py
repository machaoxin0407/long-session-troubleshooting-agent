"""Offline keyframe OCR with adapters for PaddleOCR 3 result objects."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest_io import (
    ROOT,
    project_path,
    read_csv,
    successful_preprocessing,
    write_csv,
)


DEFAULT_INPUT = ROOT / "data_video" / "manifests" / "preprocessing_manifest.csv"
DEFAULT_RUNS = ROOT / "data_video" / "manifests" / "ocr_runs.csv"
DEFAULT_OBSERVATIONS = ROOT / "data_video" / "manifests" / "ocr_observations.csv"
RUN_FIELDS = [
    "record_id",
    "status",
    "frame_count",
    "frames_with_text",
    "observation_count",
    "language_profile",
    "ocr_version",
    "device",
    "minimum_score",
    "processed_at",
    "error",
]
OBSERVATION_FIELDS = [
    "observation_id",
    "record_id",
    "frame_id",
    "timestamp_seconds",
    "frame_path",
    "text",
    "confidence",
    "polygon_json",
    "language_profile",
    "ocr_version",
]


def result_payload(result: Any) -> dict[str, Any]:
    """Normalize PaddleOCR result objects into one serializable dictionary."""
    if isinstance(result, dict):
        payload: Any = result
    else:
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
        if payload is None and hasattr(result, "to_dict"):
            payload = result.to_dict()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported PaddleOCR result type: {type(result).__name__}")
    nested = payload.get("res", payload)
    if not isinstance(nested, dict):
        raise TypeError("PaddleOCR result payload has no dictionary 'res' field")
    return nested


def polygon_json(polygon: Any) -> str:
    """Convert NumPy or Python polygon coordinates to compact JSON."""
    if polygon is None:
        return "[]"
    if hasattr(polygon, "tolist"):
        polygon = polygon.tolist()
    normalized = [[float(value) for value in point] for point in polygon]
    return json.dumps(normalized, separators=(",", ":"))


def extract_lines(result: Any, minimum_score: float) -> list[tuple[str, float, str]]:
    """Extract filtered text, confidence, and polygon triples from one result."""
    payload = result_payload(result)
    texts = list(payload.get("rec_texts") or [])
    scores = list(payload.get("rec_scores") or [])
    polygons = payload.get("rec_polys")
    if polygons is None:
        polygons = payload.get("dt_polys")
    polygons = [] if polygons is None else list(polygons)
    lines: list[tuple[str, float, str]] = []
    for index, text in enumerate(texts):
        score = float(scores[index]) if index < len(scores) else 0.0
        cleaned = str(text).strip()
        if not cleaned or score < minimum_score:
            continue
        polygon = polygons[index] if index < len(polygons) else None
        lines.append((cleaned, score, polygon_json(polygon)))
    return lines


def frame_timestamp(frame: Path, interval: float, duration: float) -> float:
    """Recover the uniform keyframe timestamp from its zero-based filename."""
    try:
        index = int(frame.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected keyframe filename: {frame.name}") from exc
    return min(index * interval, duration)


def ocr_record(
    engine: Any,
    row: dict[str, str],
    minimum_score: float,
    language_profile: str,
    ocr_version: str,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Run OCR over every uniform keyframe for one video."""
    record_id = row["record_id"]
    frame_dir = project_path(row["keyframe_dir"])
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    expected = int(row["keyframe_count"])
    if len(frames) != expected:
        raise ValueError(f"{record_id} keyframes: expected {expected}, found {len(frames)}")
    interval = float(row["keyframe_interval_seconds"])
    duration = float(row["duration_seconds"])
    observations: list[dict[str, str]] = []
    frames_with_text = 0
    for frame in frames:
        timestamp = frame_timestamp(frame, interval, duration)
        results = list(engine.predict(input=str(frame)))
        lines: list[tuple[str, float, str]] = []
        for result in results:
            lines.extend(extract_lines(result, minimum_score))
        if lines:
            frames_with_text += 1
        for text, score, polygon in lines:
            index = len(observations) + 1
            observations.append(
                {
                    "observation_id": f"{record_id}-ocr-{index:05d}",
                    "record_id": record_id,
                    "frame_id": frame.stem,
                    "timestamp_seconds": f"{timestamp:.3f}",
                    "frame_path": str(frame.relative_to(ROOT)),
                    "text": text,
                    "confidence": f"{score:.6f}",
                    "polygon_json": polygon,
                    "language_profile": language_profile,
                    "ocr_version": ocr_version,
                }
            )
    run = {
        "record_id": record_id,
        "status": "success",
        "frame_count": str(len(frames)),
        "frames_with_text": str(frames_with_text),
        "observation_count": str(len(observations)),
        "language_profile": language_profile,
        "ocr_version": ocr_version,
        "device": "",
        "minimum_score": f"{minimum_score:g}",
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "error": "",
    }
    return run, observations


def ocr_collection(
    input_path: Path = DEFAULT_INPUT,
    runs_path: Path = DEFAULT_RUNS,
    observations_path: Path = DEFAULT_OBSERVATIONS,
    language_profile: str = "de",
    ocr_version: str = "PP-OCRv5",
    device: str = "cpu",
    minimum_score: float = 0.50,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load one isolated PaddleOCR pipeline and process approved keyframes."""
    from paddleocr import PaddleOCR

    if not 0 <= minimum_score <= 1:
        raise ValueError("minimum OCR score must be between zero and one")
    engine = PaddleOCR(
        lang=language_profile,
        ocr_version=ocr_version,
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    existing_runs = (
        {row["record_id"]: row for row in read_csv(runs_path)}
        if runs_path.exists()
        else {}
    )
    existing_observations: dict[str, list[dict[str, str]]] = {}
    if observations_path.exists():
        for observation in read_csv(observations_path):
            existing_observations.setdefault(
                observation["record_id"], []
            ).append(observation)
    runs: list[dict[str, str]] = []
    all_observations: list[dict[str, str]] = []
    for row in successful_preprocessing(input_path):
        record_id = row["record_id"]
        prior_run = existing_runs.get(record_id)
        prior_observations = existing_observations.get(record_id, [])
        reusable = (
            prior_run is not None
            and prior_run["status"] == "success"
            and prior_run["language_profile"] == language_profile
            and prior_run["ocr_version"] == ocr_version
            and datetime.fromisoformat(prior_run["processed_at"])
            >= datetime.fromisoformat(row["processed_at"])
            and prior_run["minimum_score"] == f"{minimum_score:g}"
            and int(prior_run["frame_count"]) == int(row["keyframe_count"])
            and int(prior_run["observation_count"]) == len(prior_observations)
        )
        if reusable:
            print(f"ocr_skipped={record_id}", flush=True)
            runs.append(prior_run)
            all_observations.extend(prior_observations)
            write_csv(runs_path, RUN_FIELDS, runs)
            write_csv(observations_path, OBSERVATION_FIELDS, all_observations)
            continue
        print(f"ocr={record_id}", flush=True)
        try:
            run, observations = ocr_record(
                engine, row, minimum_score, language_profile, ocr_version
            )
            run["device"] = device
            all_observations.extend(observations)
        except Exception as exc:
            run = {field: "" for field in RUN_FIELDS}
            run.update(
                {
                    "record_id": record_id,
                    "status": "failed",
                    "language_profile": language_profile,
                    "ocr_version": ocr_version,
                    "device": device,
                    "minimum_score": f"{minimum_score:g}",
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        runs.append(run)
        write_csv(runs_path, RUN_FIELDS, runs)
        write_csv(observations_path, OBSERVATION_FIELDS, all_observations)
    return runs, all_observations
