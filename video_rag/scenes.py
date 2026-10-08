"""Deterministic FFmpeg scene detection and deployable clip generation."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .manifest_io import (
    ROOT,
    project_path,
    read_csv,
    successful_preprocessing,
    write_csv,
)


DEFAULT_INPUT = ROOT / "data_video" / "manifests" / "preprocessing_manifest.csv"
DEFAULT_OUTPUT = ROOT / "data_video" / "manifests" / "scene_manifest.csv"
SCENE_FIELDS = [
    "scene_id",
    "record_id",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "clip_path",
    "clip_bytes",
    "clip_sha256",
    "source_path",
    "source_sha256",
    "detector",
    "threshold",
    "minimum_scene_seconds",
    "processed_at",
    "status",
    "error",
]


def sha256_file(path: Path) -> str:
    """Hash a generated clip for reproducible evidence delivery."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_cuts(
    ffmpeg: Path, source: Path, duration: float, threshold: float, minimum: float
) -> list[float]:
    """Detect visual cuts from FFmpeg scene scores and enforce minimum spacing."""
    if not 0 < threshold < 1:
        raise ValueError("scene threshold must be between zero and one")
    if minimum <= 0:
        raise ValueError("minimum scene duration must be greater than zero")
    result = subprocess.run(
        [
            str(ffmpeg), "-hide_banner", "-loglevel", "info", "-i", str(source),
            "-filter:v", f"select='gt(scene,{threshold:g})',showinfo",
            "-an", "-f", "null", "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    candidates = sorted(
        {
            float(value)
            for value in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", result.stderr)
        }
    )
    boundaries = [0.0]
    for value in candidates:
        if value - boundaries[-1] >= minimum and duration - value >= minimum:
            boundaries.append(value)
    boundaries.append(duration)
    return boundaries


def render_clip(
    ffmpeg: Path, source: Path, destination: Path, start: float, end: float
) -> None:
    """Render one browser-compatible H.264/AAC scene clip atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.mp4")
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                str(ffmpeg), "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end - start:.3f}",
                "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
                str(temporary), "-y",
            ],
            check=True,
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def segment_record(
    row: dict[str, str], ffmpeg: Path, threshold: float, minimum: float
) -> list[dict[str, str]]:
    """Detect, render, and describe all scenes for one source video."""
    record_id = row["record_id"]
    source = project_path(row["source_path"])
    duration = float(row["duration_seconds"])
    boundaries = detect_cuts(ffmpeg, source, duration, threshold, minimum)
    scenes: list[dict[str, str]] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        scene_id = f"{record_id}-scene-{index:04d}"
        clip = ROOT / "data_video" / "processed" / record_id / "clips" / f"{scene_id}.mp4"
        render_clip(ffmpeg, source, clip, start, end)
        scenes.append(
            {
                "scene_id": scene_id,
                "record_id": record_id,
                "start_seconds": f"{start:.3f}",
                "end_seconds": f"{end:.3f}",
                "duration_seconds": f"{end - start:.3f}",
                "clip_path": str(clip.relative_to(ROOT)),
                "clip_bytes": str(clip.stat().st_size),
                "clip_sha256": sha256_file(clip),
                "source_path": row["source_path"],
                "source_sha256": row["source_sha256"],
                "detector": "ffmpeg_scene_score",
                "threshold": f"{threshold:g}",
                "minimum_scene_seconds": f"{minimum:g}",
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "status": "success",
                "error": "",
            }
        )
    return scenes


def segment_collection(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    threshold: float = 0.32,
    minimum: float = 2.0,
) -> list[dict[str, str]]:
    """Generate a scene manifest for every successfully preprocessed video."""
    import imageio_ffmpeg

    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    existing_by_record: dict[str, list[dict[str, str]]] = {}
    if output_path.exists():
        for scene in read_csv(output_path):
            existing_by_record.setdefault(scene["record_id"], []).append(scene)
    results: list[dict[str, str]] = []
    for row in successful_preprocessing(input_path):
        existing = existing_by_record.get(row["record_id"], [])
        reusable = bool(existing) and all(
            scene["status"] == "success"
            and scene["source_path"] == row["source_path"]
            and scene["source_sha256"] == row["source_sha256"]
            and scene["detector"] == "ffmpeg_scene_score"
            and scene["threshold"] == f"{threshold:g}"
            and scene["minimum_scene_seconds"] == f"{minimum:g}"
            and project_path(scene["clip_path"]).is_file()
            and project_path(scene["clip_path"]).stat().st_size
            == int(scene["clip_bytes"])
            for scene in existing
        )
        if reusable:
            print(f"segmenting_skipped={row['record_id']}", flush=True)
            results.extend(existing)
            write_csv(output_path, SCENE_FIELDS, results)
            continue
        print(f"segmenting={row['record_id']}", flush=True)
        results.extend(segment_record(row, ffmpeg, threshold, minimum))
        write_csv(output_path, SCENE_FIELDS, results)
    return results
