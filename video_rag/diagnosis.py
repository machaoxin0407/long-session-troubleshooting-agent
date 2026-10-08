"""Bounded asynchronous user-video diagnosis for the P1 product API."""

from __future__ import annotations

import base64
import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOB_ROOT = ROOT / "data_video" / "work" / "user_jobs"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".m4v"}
ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "application/octet-stream",
}


def ffmpeg_executable() -> str:
    """Use system FFmpeg when present, otherwise the pinned user-space binary."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


class VideoJobManager:
    """Persist job state on disk so a single-worker restart is inspectable."""

    def __init__(self, root: Path = DEFAULT_JOB_ROOT) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def create(self, filename: str, content_type: str, question: str, product_class: str) -> tuple[str, Path]:
        active = 0
        for state_path in self.root.glob("vjob_*/state.json"):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if state.get("status") not in {"completed", "failed"}:
                active += 1
        if active >= int(os.getenv("USER_VIDEO_MAX_ACTIVE_JOBS", "20")):
            raise ValueError("video job queue is full")
        suffix = Path(filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(f"unsupported video extension: {suffix or '(none)'}")
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise ValueError(f"unsupported content type: {content_type}")
        job_id = f"vjob_{uuid.uuid4().hex}"
        job_dir = (self.root / job_id).resolve()
        job_dir.mkdir(parents=False)
        input_path = job_dir / f"input{suffix}"
        self._write_state(
            job_id,
            {
                "job_id": job_id,
                "status": "uploading",
                "created_at": int(time.time()),
                "updated_at": int(time.time()),
                "question": question.strip(),
                "product_class": product_class.strip(),
                "input_file": input_path.name,
                "error": None,
                "result": None,
            },
        )
        return job_id, input_path

    def get(self, job_id: str) -> dict[str, Any]:
        state_path = self._state_path(job_id)
        if not state_path.is_file():
            raise KeyError(job_id)
        return json.loads(state_path.read_text(encoding="utf-8"))

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock, self._job_lock(job_id):
            state = self.get(job_id)
            state.update(changes)
            state["updated_at"] = int(time.time())
            self._atomic_write(self._state_path(job_id), state)
        return state

    def delete(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        with self._job_lock(job_id):
            state_path = job_dir / "state.json"
            if not state_path.is_file():
                raise KeyError(job_id)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") not in {"completed", "failed"}:
                raise RuntimeError("active video job cannot be deleted")
        shutil.rmtree(job_dir)

    def discard_upload(self, job_id: str) -> None:
        """Remove a job that failed before it could enter the worker queue."""
        job_dir = self._job_dir(job_id)
        state = self.get(job_id)
        if state.get("status") != "uploading":
            raise RuntimeError("only an uploading job can be discarded")
        shutil.rmtree(job_dir)

    def claim_next(self) -> str | None:
        """Atomically claim one queued job for a separate worker process."""
        with self._lock:
            for state_path in sorted(self.root.glob("vjob_*/state.json")):
                job_id = state_path.parent.name
                try:
                    with self._job_lock(job_id, wait_seconds=0):
                        state = json.loads(state_path.read_text(encoding="utf-8"))
                        lease_age = time.time() - float(state.get("updated_at") or time.time())
                        stale_claim = (
                            state.get("status") in {"claimed", "validating", "analyzing"}
                            and lease_age > float(os.getenv("USER_VIDEO_JOB_LEASE_S", "600"))
                        )
                        if state.get("status") != "queued" and not stale_claim:
                            continue
                        state["status"] = "claimed"
                        state["claimed_at"] = int(time.time())
                        state["updated_at"] = int(time.time())
                        self._atomic_write(state_path, state)
                        return str(state["job_id"])
                except (FileExistsError, OSError, json.JSONDecodeError):
                    continue
        return None

    def _job_dir(self, job_id: str) -> Path:
        if not job_id.startswith("vjob_") or not job_id[5:].isalnum():
            raise KeyError(job_id)
        path = (self.root / job_id).resolve()
        if self.root not in path.parents:
            raise KeyError(job_id)
        return path

    def _state_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "state.json"

    def _write_state(self, job_id: str, state: dict[str, Any]) -> None:
        self._atomic_write(self._state_path(job_id), state)

    @staticmethod
    def _atomic_write(path: Path, state: dict[str, Any]) -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    @contextlib.contextmanager
    def _job_lock(self, job_id: str, wait_seconds: float = 2.0):
        lock_path = self._job_dir(job_id) / ".state.lock"
        deadline = time.monotonic() + wait_seconds
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                try:
                    lock_age = time.time() - lock_path.stat().st_mtime
                    if lock_age > float(os.getenv("USER_VIDEO_LOCK_STALE_S", "300")):
                        lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        try:
            yield
        finally:
            os.close(descriptor)
            lock_path.unlink(missing_ok=True)


def probe_video(path: Path) -> dict[str, Any]:
    """Validate actual media using ffprobe, not the caller-supplied MIME type."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        import imageio_ffmpeg

        reader = imageio_ffmpeg.read_frames(str(path))
        try:
            payload = next(reader)
        finally:
            reader.close()
        width, height = payload.get("size") or (0, 0)
        duration = float(payload.get("duration") or 0)
        codec = payload.get("codec")
        return _validate_video_metadata(path, duration, int(width), int(height), codec)
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
    payload = json.loads(completed.stdout)
    streams = [stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"]
    if not streams:
        raise ValueError("uploaded file has no video stream")
    stream = streams[0]
    duration = float(payload.get("format", {}).get("duration") or stream.get("duration") or 0)
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    return _validate_video_metadata(
        path,
        duration,
        width,
        height,
        stream.get("codec_name"),
    )


def _validate_video_metadata(
    path: Path,
    duration: float,
    width: int,
    height: int,
    codec: str | None,
) -> dict[str, Any]:
    max_duration = float(os.getenv("USER_VIDEO_MAX_DURATION_S", "60"))
    max_pixels = int(os.getenv("USER_VIDEO_MAX_PIXELS", str(3840 * 2160)))
    if duration <= 0 or duration > max_duration:
        raise ValueError(f"video duration must be within (0, {max_duration}] seconds")
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise ValueError("video resolution exceeds the configured limit")
    return {
        "duration_seconds": round(duration, 3),
        "width": width,
        "height": height,
        "codec": codec,
        "size_bytes": path.stat().st_size,
    }


def extract_keyframes(
    path: Path,
    output_dir: Path,
    *,
    duration_seconds: float,
    count: int = 6,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%02d.jpg"
    command = [
        ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-vf",
        f"fps={count / max(duration_seconds, 0.001):.8f},scale='min(960,iw)':-2",
        "-frames:v",
        str(count),
        "-q:v",
        "3",
        str(pattern),
    ]
    subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    frames = sorted(output_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("no diagnostic frames were extracted")
    return frames


def _vision_diagnosis(
    frames: list[Path],
    *,
    question: str,
    product_class: str,
    standard_context: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    base_url = os.getenv("USER_VIDEO_VLM_BASE_URL", "").strip()
    api_key = os.getenv("USER_VIDEO_VLM_API_KEY", "").strip()
    model = os.getenv("USER_VIDEO_VLM_MODEL", "").strip()
    if not (base_url and api_key and model):
        return (
            {
                "label": "insufficient_evidence",
                "current_step": None,
                "deviation_type": None,
                "next_action": "配置 USER_VIDEO_VLM_* 后重新分析；当前仅完成媒体校验和关键帧提取。",
                "confidence": 0.0,
                "evidence": [],
                "safety_note": "未运行视觉模型，因此不对操作是否正确作推断。",
            },
            {"provider": "none", "model": "", "fallback": "vlm_not_configured"},
        )

    from openai import OpenAI

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "你是产品操作视频诊断器。只依据给定的有序关键帧判断，不可臆测不可见步骤。"
                "输出严格 JSON，字段为 label(correct_step|wrong_order|missing_step|wrong_component|"
                "unsafe_action|device_state_mismatch|insufficient_evidence|out_of_scope)、"
                "current_step、deviation_type、next_action、confidence(0-1)、"
                "evidence(数组，每项含 frame 与 observation)、safety_note。"
                f"\n用户问题：{question or '未提供'}\n产品类别：{product_class or '未知'}"
                f"\n可用于对齐的标准证据：\n{standard_context or '未检索到标准证据'}"
            ),
        }
    ]
    for index, frame in enumerate(frames, start=1):
        content.append({"type": "text", "text": f"Frame {index}"})
        encoded = base64.b64encode(frame.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=float(os.getenv("USER_VIDEO_VLM_TIMEOUT_S", "60")),
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
        temperature=0,
        extra_body={"enable_thinking": False},
    )
    result = json.loads(response.choices[0].message.content or "{}")
    required = {"label", "current_step", "deviation_type", "next_action", "confidence", "evidence", "safety_note"}
    if not required.issubset(result):
        raise ValueError("vision model returned an incomplete diagnosis")
    if result["label"] not in {
        "correct_step",
        "wrong_order",
        "missing_step",
        "wrong_component",
        "unsafe_action",
        "device_state_mismatch",
        "insufficient_evidence",
        "out_of_scope",
    }:
        raise ValueError("vision model returned an invalid label")
    confidence = float(result["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("vision model returned confidence outside [0, 1]")
    result["confidence"] = confidence
    if confidence < float(os.getenv("USER_VIDEO_MIN_CONFIDENCE", "0.55")):
        result.update(
            {
                "label": "insufficient_evidence",
                "deviation_type": None,
                "next_action": "证据或置信度不足，请补充更清晰且覆盖完整操作过程的视频。",
                "safety_note": "低置信度结果已自动降级，不应据此继续高风险操作。",
            }
        )
    elif result["label"] == "unsafe_action":
        result["next_action"] = (
            "立即停止当前操作并断开设备电源（仅在安全可行时）；"
            "查阅对应手册安全章节或联系合格维修人员。"
        )
        result["safety_note"] = "高风险判断采用确定性停止操作模板。"
    return result, {"provider": "openai_compatible", "model": model}


def _manual_standard_evidence(manual_engine: Any | None, query: str) -> tuple[list[dict[str, Any]], str]:
    if manual_engine is None or not query:
        return [], ""
    try:
        doc_ids = manual_engine._sparse_recall(query, top_n=4)
        results = manual_engine._build_results(doc_ids)
    except Exception:  # noqa: BLE001 - diagnosis can still use standard video
        return [], ""
    structured = [
        {
            "product": result.product,
            "heading": result.heading,
            "chunk_id": result.chunk_id,
            "excerpt": result.text[:500],
        }
        for result in results
    ]
    context = "\n".join(
        f"[MANUAL:{item['chunk_id']}] {item['product']} / {item['heading']}: {item['excerpt']}"
        for item in structured
    )
    return structured, context


def run_diagnosis_job(
    manager: VideoJobManager,
    job_id: str,
    retriever: Any | None = None,
    manual_engine: Any | None = None,
) -> None:
    """Execute a job in a background worker and always persist terminal state."""
    try:
        state = manager.update(job_id, status="validating")
        input_path = manager._job_dir(job_id) / state["input_file"]
        metadata = probe_video(input_path)
        manager.update(job_id, status="analyzing", media=metadata)
        frames = extract_keyframes(
            input_path,
            manager._job_dir(job_id) / "frames",
            duration_seconds=metadata["duration_seconds"],
        )
        query = " ".join(
            part for part in (state.get("product_class", ""), state.get("question", "")) if part
        )
        standards: list[dict[str, Any]] = []
        if retriever is not None and query:
            for item in retriever.search(query, top_k=3):
                standards.append(item.to_dict())
        manual_standards, manual_context = _manual_standard_evidence(manual_engine, query)
        video_context = "\n".join(
            f"[VIDEO:{item['scene_id']}] {item['start_seconds']}-{item['end_seconds']}s: {item['text'][:500]}"
            for item in standards
        )
        try:
            diagnosis, model_info = _vision_diagnosis(
                frames,
                question=state.get("question", ""),
                product_class=state.get("product_class", ""),
                standard_context="\n".join(part for part in (manual_context, video_context) if part),
            )
        except Exception:  # noqa: BLE001 - model outages must degrade safely
            diagnosis = {
                "label": "insufficient_evidence",
                "current_step": None,
                "deviation_type": None,
                "next_action": "视觉诊断服务暂不可用，请稍后重试或补充清晰视频。",
                "confidence": 0.0,
                "evidence": [],
                "safety_note": "模型失败已自动降级，不应据此继续高风险操作。",
            }
            model_info = {
                "provider": "openai_compatible",
                "model": os.getenv("USER_VIDEO_VLM_MODEL", ""),
                "fallback": "model_unavailable_or_invalid_output",
            }
        manager.update(
            job_id,
            status="completed",
            result={
                "diagnosis": diagnosis,
                "media": metadata,
                "sampled_frames": [f"frames/{path.name}" for path in frames],
                "matched_standard_scenes": standards,
                "matched_manual_sections": manual_standards,
                "model": model_info,
            },
        )
    except Exception as exc:  # noqa: BLE001 - background failures must be queryable
        category = "invalid_video" if isinstance(exc, ValueError) else "analysis_failed"
        manager.update(job_id, status="failed", error=category)
