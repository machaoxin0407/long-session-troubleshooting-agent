"""Auditable AI pre-annotation for frozen video relevance candidates."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest_io import ROOT
from .vlm import DEFAULT_MODEL, DEFAULT_MODEL_CACHE


DEFAULT_POOL = ROOT / "data_video" / "manifests" / "paper_relevance_pool_audit_v1.csv"
DEFAULT_OUTPUT = (
    ROOT / "data_video" / "manifests" / "paper_relevance_ai_preannotations_v1.csv"
)
DEFAULT_RUNS = (
    ROOT / "data_video" / "manifests" / "paper_relevance_ai_runs_v1.csv"
)
DEFAULT_MANIFEST = (
    ROOT / "data_video" / "manifests" / "paper_relevance_ai_run_v1.json"
)
PROMPT_VERSION = "paper-relevance-grade-json-v1"
MAX_EVIDENCE_CHARS = 8000
UNCERTAINTY_LEVELS = {"low", "medium", "high"}
EVIDENCE_SOURCES = {"VLM", "ASR", "OCR"}

LABEL_FIELDS = [
    "query_id",
    "candidate_scene_id",
    "record_id",
    "ai_relevance_grade",
    "ai_start_time",
    "ai_end_time",
    "ai_evidence",
    "ai_evidence_sources",
    "ai_uncertainty",
    "ai_uncertainty_reason",
    "model",
    "model_revision",
    "prompt_version",
    "processed_at",
]
RUN_FIELDS = [
    "pair_id",
    "query_id",
    "candidate_scene_id",
    "status",
    "model",
    "model_revision",
    "device",
    "dtype",
    "batch_size",
    "max_new_tokens",
    "prompt_version",
    "input_sha256",
    "prompt_sha256",
    "processed_at",
    "raw_output",
    "error",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pair_id(row: dict[str, str]) -> str:
    return f"{row['query_id']}|{row['candidate_scene_id']}"


def compact_evidence(text: str, maximum: int = MAX_EVIDENCE_CHARS) -> str:
    """Retain both leading ASR/OCR context and trailing VLM evidence."""
    normalized = text.strip()
    if len(normalized) <= maximum:
        return normalized
    head = maximum // 2
    tail = maximum - head
    return (
        normalized[:head].rstrip()
        + "\n[... deterministic middle truncation ...]\n"
        + normalized[-tail:].lstrip()
    )


def annotation_prompt(row: dict[str, str]) -> str:
    duration = float(row["end_seconds"]) - float(row["start_seconds"])
    evidence = compact_evidence(row["scene_text"])
    return f"""
You are an AI pre-annotator for a product-support video retrieval dataset.
Judge only the frozen evidence supplied below. Do not use outside knowledge,
retrieval rank, source title, creator identity, or an imagined missing step.

Relevance grades:
0 = unrelated to the question.
1 = same product/background only; it cannot directly answer the question.
2 = partially answers the question or shows a useful adjacent step.
3 = directly answers the question with visible, spoken, or OCR evidence.

Be conservative when evidence is incomplete. A product-name match alone is
never grade 2 or 3. For grades 0 and 1, both offsets must be null. For grades
2 and 3, give the narrowest supported offsets in seconds relative to this
scene, satisfying 0 <= start < end <= {duration:.3f}. If the evidence supports
the whole scene but not a narrower interval, use 0 and {duration:.3f}.

Return exactly one JSON object and no Markdown:
{{
  "relevance_grade": 0,
  "relevant_start_offset_seconds": null,
  "relevant_end_offset_seconds": null,
  "evidence_sources": ["VLM"],
  "evidence_text": "concise evidence-based reason",
  "uncertainty": "low",
  "uncertainty_reason": "what is missing or ambiguous"
}}

Allowed evidence_sources are VLM, ASR, and OCR. uncertainty must be low,
medium, or high.

Question ID: {row['query_id']}
Question: {row['query_text']}
Expected product class: {row['product_class']}
Question type: {row['query_type']}
Scene ID: {row['candidate_scene_id']}
Scene absolute interval: [{row['start_seconds']}, {row['end_seconds']}]

Frozen multimodal evidence:
--- evidence begins ---
{evidence}
--- evidence ends ---
""".strip()


def parse_payload(raw: str, row: dict[str, str]) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model output contains no JSON object")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model output is not a JSON object")

    grade = payload.get("relevance_grade")
    if isinstance(grade, bool) or not isinstance(grade, int) or grade not in range(4):
        raise ValueError("relevance_grade must be an integer from 0 to 3")
    uncertainty = str(payload.get("uncertainty", "")).strip().lower()
    if uncertainty not in UNCERTAINTY_LEVELS:
        raise ValueError("uncertainty must be low, medium, or high")
    reason = str(payload.get("uncertainty_reason", "")).strip()
    evidence_text = str(payload.get("evidence_text", "")).strip()
    if not evidence_text:
        raise ValueError("evidence_text must not be empty")
    sources = payload.get("evidence_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("evidence_sources must be a non-empty list")
    normalized_sources = []
    for source in sources:
        value = str(source).strip().upper()
        if value not in EVIDENCE_SOURCES:
            raise ValueError(f"Unsupported evidence source: {source}")
        if value not in normalized_sources:
            normalized_sources.append(value)

    start_offset = payload.get("relevant_start_offset_seconds")
    end_offset = payload.get("relevant_end_offset_seconds")
    scene_start = float(row["start_seconds"])
    scene_end = float(row["end_seconds"])
    duration = scene_end - scene_start
    if grade <= 1:
        if start_offset is not None or end_offset is not None:
            raise ValueError("Grades 0 and 1 must use null time offsets")
        absolute_start: str | float = ""
        absolute_end: str | float = ""
    else:
        if isinstance(start_offset, bool) or isinstance(end_offset, bool):
            raise ValueError("Time offsets must be numeric")
        if not isinstance(start_offset, (int, float)) or not isinstance(
            end_offset, (int, float)
        ):
            raise ValueError("Grades 2 and 3 require numeric time offsets")
        start_value = float(start_offset)
        end_value = float(end_offset)
        if not math.isfinite(start_value) or not math.isfinite(end_value):
            raise ValueError("Relevant offsets must be finite")
        time_basis_normalization = "relative"
        # Qwen occasionally returns the absolute source-video timestamps visible
        # in the evidence interval instead of the requested scene-relative
        # offsets. Recognize that unambiguous case before validating. The raw
        # response remains in the append-only journal for audit.
        if (
            scene_start - 0.05 <= start_value < end_value
            and end_value <= scene_end + 0.05
        ):
            start_value -= scene_start
            end_value -= scene_start
            time_basis_normalization = "absolute_to_relative"
        if (
            0 <= start_value <= duration + 0.05
            and abs(end_value - start_value) <= 0.05
        ):
            # A point timestamp cannot serve as an interval. Expand it to a
            # deterministic one-second support window, clipped to the scene.
            point = min(duration, max(0.0, start_value))
            start_value = max(0.0, point - 0.5)
            end_value = min(duration, point + 0.5)
            if end_value <= start_value:
                start_value = max(0.0, duration - 1.0)
                end_value = duration
            time_basis_normalization += "|point_to_window"
            if uncertainty == "low":
                uncertainty = "medium"
            reason = (
                f"{reason} AI returned a point timestamp; it was expanded to "
                "a clipped one-second review window."
            ).strip()
        if (
            start_value < 0
            or end_value <= start_value
            or end_value > duration + 0.05
        ):
            # The relevance evidence itself is frozen at scene level. Preserve
            # the grade but use the full scene when the model's numeric offsets
            # cannot be reconciled with either relative or absolute time.
            start_value = 0.0
            end_value = duration
            time_basis_normalization += "|invalid_to_full_scene"
            uncertainty = "high"
            reason = (
                f"{reason} AI offsets could not be reconciled with the scene; "
                "the full scene was retained for mandatory R1 review."
            ).strip()
        if (
            start_value < 0
            or end_value <= start_value
            or end_value > duration + 0.05
        ):
            raise ValueError("Relevant offsets fall outside the scene")
        absolute_start = round(scene_start + max(0.0, start_value), 3)
        absolute_end = round(scene_start + min(duration, end_value), 3)

    return {
        "query_id": row["query_id"],
        "candidate_scene_id": row["candidate_scene_id"],
        "record_id": row["record_id"],
        "ai_relevance_grade": grade,
        "ai_start_time": absolute_start,
        "ai_end_time": absolute_end,
        "ai_evidence": evidence_text,
        "ai_evidence_sources": "|".join(normalized_sources),
        "ai_uncertainty": uncertainty,
        "ai_uncertainty_reason": reason,
        "time_basis_normalization": (
            time_basis_normalization if grade >= 2 else "not_applicable"
        ),
    }


def load_journal(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid journal JSON at line {line_number}") from exc
            latest[row["pair_id"]] = row
    return latest


def append_journal(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


def generate_text(model: Any, processor: Any, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
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
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    trimmed = [
        output[len(source) :] for source, output in zip(inputs.input_ids, generated)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def generate_batch_text(
    model: Any,
    processor: Any,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    """Generate one response per text prompt with fixed padded batching."""
    conversations = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        for prompt in prompts
    ]
    texts = [
        processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        for conversation in conversations
    ]
    tokenizer = processor.tokenizer
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = processor(
            text=texts,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    finally:
        tokenizer.padding_side = original_padding_side
    input_length = inputs.input_ids.shape[1]
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    trimmed = generated[:, input_length:]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def annotate_shard(
    *,
    pool_path: Path,
    journal_path: Path,
    shard_index: int,
    shard_count: int,
    model_name: str = DEFAULT_MODEL,
    model_cache: Path = DEFAULT_MODEL_CACHE,
    device: str,
    dtype: str = "bfloat16",
    offline: bool = True,
    limit: int | None = None,
    max_new_tokens: int = 220,
    batch_size: int = 8,
) -> dict[str, int]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard selection")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = read_csv(pool_path)
    identifiers = [pair_id(row) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Candidate pool contains duplicate query-scene pairs")
    selected = [
        row
        for index, row in enumerate(rows)
        if index % shard_count == shard_index
    ]
    if limit is not None:
        selected = selected[:limit]
    existing = load_journal(journal_path)
    pending = [
        row
        for row in selected
        if existing.get(pair_id(row), {}).get("status") != "success"
    ]
    if not pending:
        return {"selected": len(selected), "completed": len(selected), "failed": 0}

    dtype_value = getattr(torch, dtype)
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
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        prompts = [annotation_prompt(row) for row in batch]
        print(
            f"ai_relevance shard={shard_index} "
            f"batch={batch_start // batch_size + 1}/"
            f"{math.ceil(len(pending) / batch_size)} "
            f"items={batch_start + 1}-{batch_start + len(batch)}/{len(pending)}",
            flush=True,
        )
        try:
            raw_outputs = generate_batch_text(
                model,
                processor,
                prompts,
                max_new_tokens,
            )
            if len(raw_outputs) != len(batch):
                raise ValueError("Batch generation returned the wrong response count")
            batch_errors: list[Exception | None] = [None] * len(batch)
        except Exception as exc:  # noqa: BLE001 - retry each item independently
            raw_outputs = [""] * len(batch)
            batch_errors = [exc] * len(batch)
        for row, prompt, raw, batch_error in zip(
            batch, prompts, raw_outputs, batch_errors
        ):
            identifier = pair_id(row)
            now = datetime.now(timezone.utc).isoformat()
            base = {
                "pair_id": identifier,
                "query_id": row["query_id"],
                "candidate_scene_id": row["candidate_scene_id"],
                "model": model_name,
                "model_revision": revision,
                "device": device,
                "dtype": dtype,
                "batch_size": str(batch_size),
                "max_new_tokens": str(max_new_tokens),
                "prompt_version": PROMPT_VERSION,
                "input_sha256": sha256_text(
                    json.dumps(
                        {
                            "query_id": row["query_id"],
                            "query_text": row["query_text"],
                            "product_class": row["product_class"],
                            "query_type": row["query_type"],
                            "candidate_scene_id": row["candidate_scene_id"],
                            "start_seconds": row["start_seconds"],
                            "end_seconds": row["end_seconds"],
                            "scene_text": compact_evidence(row["scene_text"]),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
                "prompt_sha256": sha256_text(prompt),
                "processed_at": now,
            }
            try:
                if batch_error is not None:
                    raw = generate_text(model, processor, prompt, max_new_tokens)
                try:
                    label = parse_payload(raw, row)
                except (ValueError, json.JSONDecodeError):
                    repair_prompt = (
                        prompt
                        + "\n\nYour previous response was invalid. Return one corrected "
                        "JSON object only, obeying the grade and time-offset rules."
                    )
                    raw = generate_text(
                        model, processor, repair_prompt, max_new_tokens
                    )
                    label = parse_payload(raw, row)
                label.update(
                    {
                        "model": model_name,
                        "model_revision": revision,
                        "prompt_version": PROMPT_VERSION,
                        "processed_at": now,
                    }
                )
                record = {
                    **base,
                    "status": "success",
                    "raw_output": raw,
                    "error": "",
                }
                record["label"] = label
            except Exception as exc:  # noqa: BLE001 - retain failure for retry
                record = {
                    **base,
                    "status": "failed",
                    "raw_output": raw,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            append_journal(journal_path, record)
    selected_ids = {pair_id(row) for row in selected}
    completed = sum(
        value.get("status") == "success"
        for value in load_journal(journal_path).values()
        if value.get("pair_id") in selected_ids
    )
    return {
        "selected": len(selected),
        "completed": completed,
        "failed": len(selected) - completed,
    }


def merge_journals(
    *,
    pool_path: Path,
    journals: list[Path],
    labels_path: Path = DEFAULT_OUTPUT,
    runs_path: Path = DEFAULT_RUNS,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    pool_path = pool_path.resolve()
    journals = [path.resolve() for path in journals]
    labels_path = labels_path.resolve()
    runs_path = runs_path.resolve()
    manifest_path = manifest_path.resolve()
    pool = read_csv(pool_path)
    expected = {pair_id(row): row for row in pool}
    merged: dict[str, dict[str, Any]] = {}
    for journal in journals:
        for identifier, row in load_journal(journal).items():
            if identifier in merged:
                raise ValueError(f"Pair occurs in multiple journals: {identifier}")
            merged[identifier] = row
    missing = sorted(set(expected) - set(merged))
    extra = sorted(set(merged) - set(expected))
    failed = sorted(
        identifier
        for identifier, row in merged.items()
        if row.get("status") != "success"
    )
    if missing or extra or failed:
        raise ValueError(
            f"Incomplete AI labels: missing={missing[:10]} extra={extra[:10]} "
            f"failed={failed[:10]}"
        )
    signatures = {
        (
            row["model"],
            row["model_revision"],
            row["dtype"],
            row["batch_size"],
            row["max_new_tokens"],
            row["prompt_version"],
        )
        for row in merged.values()
    }
    if len(signatures) != 1:
        raise ValueError(f"AI run signatures differ: {sorted(signatures)}")
    ordered_ids = [pair_id(row) for row in pool]
    labels = [merged[identifier]["label"] for identifier in ordered_ids]
    runs = [
        {field: merged[identifier].get(field, "") for field in RUN_FIELDS}
        for identifier in ordered_ids
    ]
    write_csv(labels_path, LABEL_FIELDS, labels)
    write_csv(runs_path, RUN_FIELDS, runs)
    grades: dict[str, int] = {
        str(grade): sum(int(row["ai_relevance_grade"]) == grade for row in labels)
        for grade in range(4)
    }
    signature = next(iter(signatures))
    manifest = {
        "schema_version": "paper-relevance-ai-preannotation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pool_path": str(pool_path.relative_to(ROOT)).replace("\\", "/"),
        "pool_sha256": sha256_file(pool_path),
        "pair_count": len(labels),
        "grade_counts": grades,
        "model": signature[0],
        "model_revision": signature[1],
        "dtype": signature[2],
        "batch_size": int(signature[3]),
        "prompt_version": signature[5],
        "deterministic_generation": {
            "do_sample": False,
            "max_new_tokens": int(signature[4]),
        },
        "evidence_compaction": {
            "maximum_characters": MAX_EVIDENCE_CHARS,
            "strategy": "full text or deterministic head-tail truncation",
        },
        "time_boundary_policy": {
            "grades_0_1": "blank",
            "grades_2_3": "absolute interval constrained to candidate scene",
            "normalization": (
                "absolute timestamps are converted to scene-relative offsets; "
                "point timestamps become clipped one-second windows; "
                "irreconcilable offsets retain the full scene and high uncertainty"
            ),
        },
        "journals": [str(path.relative_to(ROOT)).replace("\\", "/") for path in journals],
        "outputs": {
            "labels": str(labels_path.relative_to(ROOT)).replace("\\", "/"),
            "runs": str(runs_path.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
