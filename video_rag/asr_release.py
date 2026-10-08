"""Replace overlong fused narrative documents with timed transcript chunks."""

import hashlib
import json
import math

from .evidence import interval_overlap, unique_text


def render_speech(segments: list[dict[str, str]]) -> str:
    header = ('ASR: Timed transcript excerpt. Times refer to the source video; a segment '
              'may overlap the clip boundary. Segment inclusion alone does not establish '
              'procedure completeness, device identity, or success.')
    views = [{'start_seconds': float(s['start_seconds']), 'end_seconds': float(s['end_seconds']),
              'text': s['text']} for s in segments]
    return header + '\n' + '\n'.join(json.dumps(v, ensure_ascii=False) for v in views)


def expand_long_speech(evidence: list[dict[str, str]], segments: list[dict[str, str]],
                       max_chars: int = 2000) -> list[dict[str, str]]:
    """Preserve whole source segments and reject missing or mismatched provenance."""
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError('Speech budget must be a positive integer')
    by_id = {s['segment_id']: s for s in segments}
    if len(by_id) != len(segments):
        raise ValueError('Duplicate ASR segment identity')
    used = {r['evidence_id'] for r in evidence}
    if len(used) != len(evidence):
        raise ValueError('Duplicate evidence identity')
    output = []
    for parent in evidence:
        if parent['evidence_type'] != 'video_scene' or len(parent['text']) <= max_chars:
            output.append(dict(parent))
            continue
        if parent.get('ocr_observation_ids') or not parent.get('asr_segment_ids'):
            raise ValueError('Overlong source is not an isolated ASR/VLM narrative')
        ids = parent['asr_segment_ids'].split('|')
        if len(ids) != len(set(ids)):
            raise ValueError('Repeated ASR identity in parent')
        try:
            selected = [by_id[key] for key in ids]
        except KeyError as exc:
            raise ValueError('Missing ASR segment') from exc
        start, end = float(parent['start_seconds']), float(parent['end_seconds'])
        for segment in selected:
            a, b = float(segment['start_seconds']), float(segment['end_seconds'])
            if (not math.isfinite(a) or not math.isfinite(b) or not 0 <= a < b
                    or segment['record_id'] != parent['record_id']
                    or not interval_overlap(segment, start, end)):
                raise ValueError('ASR segment does not belong to parent interval')
        parts = parent['text'].split('\nVLM:', 1)
        if parts[0] != 'ASR: ' + unique_text(selected):
            raise ValueError('ASR segments do not reconstruct parent text')
        visual = 'VLM:' + parts[1] if len(parts) == 2 else ''
        if len(visual) > max_chars:
            raise ValueError('Complete visual summary exceeds the budget')
        # Sorting changes only presentation; the reconstructed original above
        # must match the frozen source before any reordering is permitted.
        selected = sorted(selected, key=lambda s: (float(s['start_seconds']), float(s['end_seconds']), s['segment_id']))
        chunks, pending = [], []
        for segment in selected:
            if len(render_speech([segment])) > max_chars:
                raise ValueError('Complete ASR segment exceeds the budget')
            if pending and len(render_speech([*pending, segment])) > max_chars:
                chunks.append(pending)
                pending = []
            pending.append(segment)
        if pending:
            chunks.append(pending)
        meta = json.loads(parent.get('metadata_json') or '{}')
        provenance = {**meta, 'parent_scene_id': meta.get('parent_scene_id', parent['evidence_id']),
                      'parent_document_id': parent['evidence_id'],
                      'parent_document_text_sha256': hashlib.sha256(parent['text'].encode()).hexdigest(),
                      'media_interval_scope': 'complete_parent_clip_not_transcript_duration'}
        output.append(dict(parent, evidence_type='video_scene_parent'))
        children = []
        if visual:
            children.append(dict(parent, evidence_id=parent['evidence_id'] + '-vlm', text=visual,
                                 asr_segment_ids='', language='', confidence='',
                                 metadata_json=json.dumps({**provenance, 'evidence_scope': 'visual_summary_only'}, sort_keys=True)))
        for i, chunk in enumerate(chunks, 1):
            children.append(dict(parent, evidence_id=f"{parent['evidence_id']}-asr-{i:04d}",
                                 text=render_speech(chunk), vlm_caption_ids='', confidence='',
                                 asr_segment_ids='|'.join(s['segment_id'] for s in chunk),
                                 metadata_json=json.dumps({**provenance, 'evidence_scope': 'timed_asr_segments_only',
                                                           'asr_segments': chunk,
                                                           'asr_chunk_index': i, 'asr_chunk_count': len(chunks)}, sort_keys=True)))
        for child in children:
            if child['evidence_id'] in used:
                raise ValueError('Derived evidence identity collision')
            used.add(child['evidence_id'])
            output.append(child)
    return output
