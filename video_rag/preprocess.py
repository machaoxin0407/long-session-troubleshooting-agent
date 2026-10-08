"""Review-gated, auditable preprocessing for authorized source videos."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPTS = ROOT / "data_video" / "manifests" / "download_receipts.csv"
DEFAULT_REVIEWS = ROOT / "data_video" / "manifests" / "review_assignments.csv"
DEFAULT_MANIFEST = ROOT / "data_video" / "manifests" / "preprocessing_manifest.csv"
DEFAULT_PROCESSED = ROOT / "data_video" / "processed"
DEFAULT_KEYFRAMES = ROOT / "data_video" / "keyframes"
MANIFEST_FIELDS = [
    "record_id",
    "source_path",
    "source_sha256",
    "source_variant",
    "processed_at",
    "status",
    "duration_seconds",
    "fps",
    "width",
    "height",
    "video_codec",
    "pixel_format",
    "audio_status",
    "audio_path",
    "keyframe_interval_seconds",
    "keyframe_count",
    "keyframe_dir",
    "ffmpeg_version",
    "error",
]


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading a video into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one UTF-8 CSV into plain dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def accepted_record_ids(review_path: Path) -> set[str]:
    """Return records accepted by either the legacy or current review gate."""
    accepted: set[str] = set()
    for row in read_csv(review_path):
        legacy_complete = (
            row.get("license_review_status") == "approved"
            and row.get("content_review_status") == "approved"
            and row.get("adjudication_status") == "approved"
        )
        single_reviewer_complete = (
            row.get("review_protocol") == "ai_single_reviewer_v1"
            and row.get("reviewer_id") == "R1"
            and row.get("review_status") == "completed"
            and row.get("license_evidence_status") == "verified"
        )
        if (legacy_complete or single_reviewer_complete) and row.get(
            "final_decision"
        ) == "accept":
            accepted.add(row["record_id"])
    return accepted


def resolve_project_path(relative_path: str) -> Path:
    """Resolve a receipt path and reject paths that escape the repository."""
    candidate = (ROOT / relative_path).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes project root: {relative_path}") from exc
    return candidate


def video_metadata(source: Path, ffmpeg: Path) -> dict[str, Any]:
    """Read video metadata and recover frame rate from FFmpeg when needed."""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(source), pix_fmt="rgb24")
    try:
        metadata = next(reader)
    finally:
        reader.close()
    if float(metadata.get("fps") or 0) <= 0:
        result = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-i", str(source)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        video_line = next(
            (line for line in result.stderr.splitlines() if "Video:" in line), ""
        )
        rate = re.search(r"(\d+(?:\.\d+)?)\s+(?:fps|tbr)\b", video_line)
        if rate:
            metadata["fps"] = float(rate.group(1))
    return metadata


def has_audio_stream(ffmpeg: Path, source: Path) -> bool:
    """Inspect FFmpeg stream descriptions without decoding the full input."""
    result = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-i", str(source)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return bool(re.search(r"Stream #.*: Audio:", result.stderr))


def extract_audio(ffmpeg: Path, source: Path, destination: Path) -> str:
    """Extract mono 16 kHz PCM audio, or report that no audio stream exists."""
    if not has_audio_stream(ffmpeg, source):
        destination.unlink(missing_ok=True)
        return "absent"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.wav")
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(source),
                "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(temporary), "-y",
            ],
            check=True,
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "extracted"


def extract_keyframes(
    ffmpeg: Path, source: Path, destination: Path, interval_seconds: float
) -> int:
    """Write deterministic uniformly sampled JPEG evidence frames."""
    if interval_seconds <= 0:
        raise ValueError("keyframe interval must be greater than zero")
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        subprocess.run(
            [
                str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(source),
                "-vf", f"fps=1/{interval_seconds:g}", "-q:v", "2", "-start_number", "0",
                str(temporary / "frame_%06d.jpg"), "-y",
            ],
            check=True,
        )
        frames = sorted(temporary.glob("frame_*.jpg"))
        if not frames:
            raise RuntimeError("FFmpeg produced no keyframes")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        return len(frames)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Atomically write preprocessing results in stable record order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["record_id"]))
    os.replace(temporary, path)


def reusable_preprocessing_row(
    row: dict[str, str], receipt: dict[str, str], interval_seconds: float
) -> bool:
    """Return whether a successful prior row and its artifacts remain reusable."""
    if row.get("status") != "success":
        return False
    if row.get("source_sha256") != receipt.get("sha256"):
        return False
    try:
        prior_interval = float(row.get("keyframe_interval_seconds", ""))
    except ValueError:
        return False
    if prior_interval != interval_seconds:
        return False
    frame_dir = resolve_project_path(row.get("keyframe_dir", ""))
    if not frame_dir.is_dir() or not next(frame_dir.glob("frame_*.jpg"), None):
        return False
    if row.get("audio_status") == "extracted":
        audio_path = resolve_project_path(row.get("audio_path", ""))
        if not audio_path.is_file():
            return False
    return True


def preprocess_one(
    receipt: dict[str, str], ffmpeg: Path, ffmpeg_version: str, interval_seconds: float
) -> dict[str, str]:
    """Verify and preprocess one accepted source into audio and keyframes."""
    record_id = receipt["record_id"]
    source = resolve_project_path(receipt["local_path"])
    source_digest = sha256_file(source)
    if source_digest != receipt["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {record_id}")

    metadata = video_metadata(source, ffmpeg)
    size = metadata.get("size") or metadata.get("source_size") or (0, 0)
    audio_path = DEFAULT_PROCESSED / record_id / "audio_16k_mono.wav"
    frame_dir = DEFAULT_KEYFRAMES / record_id
    audio_status = extract_audio(ffmpeg, source, audio_path)
    frame_count = extract_keyframes(ffmpeg, source, frame_dir, interval_seconds)
    return {
        "record_id": record_id,
        "source_path": str(source.relative_to(ROOT)),
        "source_sha256": source_digest,
        "source_variant": receipt.get("source_variant", ""),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "duration_seconds": str(metadata.get("duration", "")),
        "fps": str(metadata.get("fps", "")),
        "width": str(size[0]),
        "height": str(size[1]),
        "video_codec": str(metadata.get("codec", "")).rstrip(","),
        "pixel_format": str(metadata.get("pix_fmt", "")),
        "audio_status": audio_status,
        "audio_path": str(audio_path.relative_to(ROOT)) if audio_status == "extracted" else "",
        "keyframe_interval_seconds": f"{interval_seconds:g}",
        "keyframe_count": str(frame_count),
        "keyframe_dir": str(frame_dir.relative_to(ROOT)),
        "ffmpeg_version": ffmpeg_version,
        "error": "",
    }


def preprocess_collection(
    receipts_path: Path = DEFAULT_RECEIPTS,
    reviews_path: Path = DEFAULT_REVIEWS,
    manifest_path: Path = DEFAULT_MANIFEST,
    interval_seconds: float = 5.0,
) -> list[dict[str, str]]:
    """Preprocess every downloaded record that passed a supported review gate."""
    import imageio_ffmpeg

    accepted = accepted_record_ids(reviews_path)
    receipts = [row for row in read_csv(receipts_path) if row["record_id"] in accepted]
    receipt_ids = {row["record_id"] for row in receipts}
    missing = sorted(accepted - receipt_ids)
    if missing:
        raise ValueError(f"Accepted records have no download receipt: {', '.join(missing)}")

    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    ffmpeg_version = imageio_ffmpeg.get_ffmpeg_version()
    prior = (
        {row["record_id"]: row for row in read_csv(manifest_path)}
        if manifest_path.exists()
        else {}
    )
    rows: list[dict[str, str]] = []
    for receipt in sorted(receipts, key=lambda row: row["record_id"]):
        record_id = receipt["record_id"]
        prior_row = prior.get(record_id)
        if prior_row and reusable_preprocessing_row(
            prior_row, receipt, interval_seconds
        ):
            rows.append(prior_row)
            print(f"preprocessing_skipped={record_id}", flush=True)
            continue
        print(f"preprocessing={record_id}", flush=True)
        try:
            row = preprocess_one(receipt, ffmpeg, ffmpeg_version, interval_seconds)
        except Exception as exc:
            row = {field: "" for field in MANIFEST_FIELDS}
            row.update(
                {
                    "record_id": record_id,
                    "source_path": receipt.get("local_path", ""),
                    "source_sha256": receipt.get("sha256", ""),
                    "source_variant": receipt.get("source_variant", ""),
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "keyframe_interval_seconds": f"{interval_seconds:g}",
                    "ffmpeg_version": ffmpeg_version,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        rows.append(row)
        write_manifest(manifest_path, rows)
    # The final accepted records may all reuse prior artifacts. Persist once
    # more after the loop so a trailing run of skipped rows is not omitted.
    write_manifest(manifest_path, rows)
    return rows
