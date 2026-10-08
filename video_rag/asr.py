"""Offline multilingual ASR with segment and word-level timestamps."""

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
DEFAULT_RUNS = ROOT / "data_video" / "manifests" / "asr_runs.csv"
DEFAULT_SEGMENTS = ROOT / "data_video" / "manifests" / "asr_segments.csv"
DEFAULT_MODEL_CACHE = ROOT / "models" / "faster-whisper"
RUN_FIELDS = [
    "record_id",
    "status",
    "language",
    "language_probability",
    "audio_duration_seconds",
    "segment_count",
    "word_count",
    "model",
    "device",
    "device_index",
    "compute_type",
    "processed_at",
    "error",
]
SEGMENT_FIELDS = [
    "segment_id",
    "record_id",
    "start_seconds",
    "end_seconds",
    "text",
    "language",
    "language_probability",
    "avg_log_probability",
    "no_speech_probability",
    "compression_ratio",
    "words_json",
    "model",
]


def serialize_words(words: list[Any] | None) -> tuple[str, int]:
    """Serialize optional word timestamps without depending on model classes."""
    payload: list[dict[str, object]] = []
    for word in words or []:
        payload.append(
            {
                "start": round(float(word.start), 3),
                "end": round(float(word.end), 3),
                "word": str(word.word),
                "probability": round(float(word.probability), 6),
            }
        )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), len(payload)


def transcribe_record(
    model: Any, row: dict[str, str], model_name: str, beam_size: int
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Transcribe one preprocessed audio file and return run and segment rows."""
    record_id = row["record_id"]
    if row.get("audio_status") != "extracted" or not row.get("audio_path"):
        # A video without an audio stream is a valid zero-evidence ASR result,
        # not a pipeline failure.  Recording it as success keeps full source
        # coverage auditable while producing no speech segments.
        return (
            {
                "record_id": record_id,
                "status": "success",
                "language": "none",
                "language_probability": "0.000000",
                "audio_duration_seconds": "0.000",
                "segment_count": "0",
                "word_count": "0",
                "model": model_name,
                "device": "",
                "device_index": "",
                "compute_type": "",
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "error": "",
            },
            [],
        )
    audio = project_path(row["audio_path"])
    generated, info = model.transcribe(
        str(audio),
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    segments: list[dict[str, str]] = []
    total_words = 0
    for index, segment in enumerate(generated, start=1):
        words_json, word_count = serialize_words(segment.words)
        total_words += word_count
        segments.append(
            {
                "segment_id": f"{record_id}-asr-{index:04d}",
                "record_id": record_id,
                "start_seconds": f"{float(segment.start):.3f}",
                "end_seconds": f"{float(segment.end):.3f}",
                "text": str(segment.text).strip(),
                "language": str(info.language),
                "language_probability": f"{float(info.language_probability):.6f}",
                "avg_log_probability": f"{float(segment.avg_logprob):.6f}",
                "no_speech_probability": f"{float(segment.no_speech_prob):.6f}",
                "compression_ratio": f"{float(segment.compression_ratio):.6f}",
                "words_json": words_json,
                "model": model_name,
            }
        )
    run = {
        "record_id": record_id,
        "status": "success",
        "language": str(info.language),
        "language_probability": f"{float(info.language_probability):.6f}",
        "audio_duration_seconds": f"{float(info.duration):.3f}",
        "segment_count": str(len(segments)),
        "word_count": str(total_words),
        "model": model_name,
        "device": "",
        "device_index": "",
        "compute_type": "",
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "error": "",
    }
    return run, segments


def transcribe_collection(
    input_path: Path = DEFAULT_INPUT,
    runs_path: Path = DEFAULT_RUNS,
    segments_path: Path = DEFAULT_SEGMENTS,
    model_name: str = "large-v3",
    model_cache: Path = DEFAULT_MODEL_CACHE,
    device: str = "cuda",
    device_index: int = 0,
    compute_type: str = "float16",
    beam_size: int = 5,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load one Whisper model and transcribe all manifest-approved audio."""
    from faster_whisper import WhisperModel

    model_cache.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(
        model_name,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        download_root=str(model_cache),
    )
    existing_runs = (
        {row["record_id"]: row for row in read_csv(runs_path)}
        if runs_path.exists()
        else {}
    )
    existing_segments: dict[str, list[dict[str, str]]] = {}
    if segments_path.exists():
        for segment in read_csv(segments_path):
            existing_segments.setdefault(segment["record_id"], []).append(segment)
    runs: list[dict[str, str]] = []
    all_segments: list[dict[str, str]] = []
    for row in successful_preprocessing(input_path):
        record_id = row["record_id"]
        prior_run = existing_runs.get(record_id)
        prior_segments = existing_segments.get(record_id, [])
        reusable = (
            prior_run is not None
            and prior_run["status"] == "success"
            and prior_run["model"] == model_name
            and datetime.fromisoformat(prior_run["processed_at"])
            >= datetime.fromisoformat(row["processed_at"])
            and int(prior_run["segment_count"]) == len(prior_segments)
            and all(segment["model"] == model_name for segment in prior_segments)
        )
        if reusable:
            print(f"transcribing_skipped={record_id}", flush=True)
            runs.append(prior_run)
            all_segments.extend(prior_segments)
            write_csv(runs_path, RUN_FIELDS, runs)
            write_csv(segments_path, SEGMENT_FIELDS, all_segments)
            continue
        print(f"transcribing={record_id}", flush=True)
        try:
            run, segments = transcribe_record(model, row, model_name, beam_size)
            run.update(
                {
                    "device": device,
                    "device_index": str(device_index),
                    "compute_type": compute_type,
                }
            )
            all_segments.extend(segments)
        except Exception as exc:
            run = {field: "" for field in RUN_FIELDS}
            run.update(
                {
                    "record_id": record_id,
                    "status": "failed",
                    "model": model_name,
                    "device": device,
                    "device_index": str(device_index),
                    "compute_type": compute_type,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        runs.append(run)
        write_csv(runs_path, RUN_FIELDS, runs)
        write_csv(segments_path, SEGMENT_FIELDS, all_segments)
    return runs, all_segments
