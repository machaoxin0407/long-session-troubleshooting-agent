"""Shared manifest utilities for review-gated video analysis stages."""

from __future__ import annotations

import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV manifest into plain dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    """Atomically write rows using an explicit stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def project_path(relative_path: str) -> Path:
    """Resolve a project-relative path while rejecting directory traversal."""
    # Manifests may be produced on Windows and consumed on Linux (or vice
    # versa), so normalize both separator styles before resolving.
    candidate = (ROOT / relative_path.replace("\\", "/")).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes project root: {relative_path}") from exc
    return candidate


def successful_preprocessing(path: Path) -> list[dict[str, str]]:
    """Return only successful records from the preprocessing manifest."""
    rows = [row for row in read_csv(path) if row.get("status") == "success"]
    if not rows:
        raise ValueError(f"No successful preprocessing rows in {path}")
    return rows
