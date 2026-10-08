"""Preserve OCR sampling context without inventing an instruction sequence."""

import json
import math


def frame_groups(rows: list[dict[str, str]]) -> list[dict]:
    """Keep every observation, grouping simultaneous text by sampled frame."""
    groups = {}
    seen = set()
    for row in rows:
        timestamp = float(row['timestamp_seconds'])
        frame = row['frame_id']
        observation = row['observation_id']
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('OCR timestamp must be finite and nonnegative')
        if not frame or not observation or observation in seen:
            raise ValueError('OCR needs frame identities and unique observation identities')
        seen.add(observation)
        key = (timestamp, frame)
        groups.setdefault(key, []).append({'observation_id': observation, 'text': row['text']})
    return [{'timestamp_seconds': timestamp, 'frame_id': frame, 'observations': observations}
            for (timestamp, frame), observations in sorted(groups.items())]


def render_timeline(groups: list[dict]) -> str:
    """Render sampled text as observations, not continuous actions or steps."""
    if not groups:
        return ''
    header = ('OCR: Sampled-frame text; entries within a frame are simultaneous '
              'observations, not action order. Repeated text at later timestamps '
              'is retained. Numbered parts contain only part of the SAME frame, '
              'not successive actions. Frame sampling does not verify intervening actions, '
              'button counts, procedure completeness, or success.')
    # Observation identities remain in structured groups and the release's
    # ocr_observation_ids, in the same order. Avoid repeating them in model text.
    views = [{'timestamp_seconds': group['timestamp_seconds'], 'frame_id': group['frame_id'],
              **{key: group[key] for key in ('part_index', 'part_count') if key in group},
              'texts': [o['text'] for o in group['observations']]} for group in groups]
    return header + '\n' + '\n'.join(json.dumps(view, ensure_ascii=False) for view in views)


def _frame_parts(group: dict, max_chars: int) -> list[dict]:
    if len(render_timeline([group])) <= max_chars:
        return [group]
    observations = group['observations']
    # Reserve enough digits for the worst case (one observation per part).
    template = {**group, 'part_index': len(observations), 'part_count': len(observations)}
    parts, pending = [], []
    for observation in observations:
        if len(render_timeline([{**template, 'observations': [observation]}])) > max_chars:
            raise ValueError('A complete OCR observation exceeds the timeline budget')
        if pending and len(render_timeline([{**template, 'observations': [*pending, observation]}])) > max_chars:
            parts.append({**group, 'observations': pending})
            pending = []
        pending.append(observation)
    if pending:
        parts.append({**group, 'observations': pending})
    return [{**part, 'part_index': i, 'part_count': len(parts)} for i, part in enumerate(parts, 1)]


def timeline_chunks(groups: list[dict], max_chars: int = 2000) -> list[dict]:
    """Prefer whole frames; split oversized frames only at observation boundaries.

    These are source chunks, not selected retrieval results. Every returned
    chunk retains the sampling warning and its exact observation identities.
    """
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError('Timeline budget must be a positive integer')
    chunks = []
    pending = []
    for group in [part for frame in groups for part in _frame_parts(frame, max_chars)]:
        if pending and len(render_timeline([*pending, group])) > max_chars:
            chunks.append({'frames': pending, 'text': render_timeline(pending)})
            pending = []
        pending.append(group)
    if pending:
        chunks.append({'frames': pending, 'text': render_timeline(pending)})
    return chunks
