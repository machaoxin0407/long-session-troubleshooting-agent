"""Build new retrieval documents from whole OCR frames, without editing inputs."""

import hashlib
import json

from .evidence import timestamp_inside, unique_text
from .ocr_timeline import frame_groups, render_timeline, timeline_chunks


def expand_ocr_scenes(evidence: list[dict[str, str]], observations: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep parent media coordinates while indexing independently citable chunks.

    A returned scene interval still identifies the complete parent clip, not an
    invented continuous OCR observation. Sample times live in each chunk's text.
    Dense and visual indexes must be rebuilt for the new manifest identity.
    """
    by_id = {}
    for row in observations:
        key = row['observation_id']
        if key in by_id:
            raise ValueError('Duplicate upstream OCR observation identity')
        by_id[key] = row
    original_ids = [row['evidence_id'] for row in evidence]
    if len(set(original_ids)) != len(original_ids):
        raise ValueError('Duplicate evidence identity')
    used_ids = set(original_ids)
    output = []
    for parent in evidence:
        if parent['evidence_type'] != 'video_scene' or not parent.get('ocr_observation_ids'):
            output.append(dict(parent))
            continue
        ids = parent['ocr_observation_ids'].split('|')
        try:
            rows = [by_id[key] for key in ids]
        except KeyError as exc:
            raise ValueError('Missing upstream OCR observation') from exc
        start, end = float(parent['start_seconds']), float(parent['end_seconds'])
        if any(row['record_id'] != parent['record_id'] or not timestamp_inside(row, start, end) for row in rows):
            raise ValueError('OCR observation does not belong to parent scene')
        groups = frame_groups(rows)
        chunks = timeline_chunks(groups)
        # Verify the upstream rows actually reconstruct the frozen source before
        # replacing its representation. Do not trust IDs alone.
        old_ocr = 'OCR: ' + unique_text(rows)
        new_ocr = render_timeline(groups)
        text = parent['text']
        position = text.find('OCR: ') if text.startswith('OCR: ') else text.find('\nOCR: ')
        if position < 0:
            raise ValueError('Parent text has no OCR block')
        if text[position] == '\n':
            position += 1
        suffix = text.find('\nVLM:', position)
        suffix = len(text) if suffix < 0 else suffix
        if text[position:suffix] not in (old_ocr, new_ocr):
            raise ValueError('OCR rows do not reconstruct parent text')
        narrative = (text[:position].rstrip('\n') + '\n' + text[suffix:].lstrip('\n')).strip('\n')
        meta = json.loads(parent.get('metadata_json') or '{}')
        provenance = {'parent_scene_id': parent['evidence_id'],
                      'parent_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                      'media_interval_scope': 'complete_parent_clip_not_ocr_duration'}
        # Preserve the complete original row for audits, but exclude it from
        # retrievers which explicitly select evidence_type=video_scene.
        archived = dict(parent, evidence_type='video_scene_parent')
        output.append(archived)
        if narrative:
            key = parent['evidence_id'] + '-narrative'
            if key in used_ids:
                raise ValueError('Derived evidence identity collision')
            used_ids.add(key)
            output.append(dict(parent, evidence_id=key, text=narrative, ocr_observation_ids='',
                               metadata_json=json.dumps({**meta, **provenance,
                                                         'evidence_scope': 'parent_asr_vlm_only'}, sort_keys=True)))
        for index, chunk in enumerate(chunks, 1):
            key = f"{parent['evidence_id']}-ocr-{index:04d}"
            if key in used_ids:
                raise ValueError('Derived evidence identity collision')
            used_ids.add(key)
            chunk_ids = [o['observation_id'] for frame in chunk['frames'] for o in frame['observations']]
            output.append(dict(parent, evidence_id=key, text=chunk['text'],
                               asr_segment_ids='', vlm_caption_ids='', language='', confidence='',
                               ocr_observation_ids='|'.join(chunk_ids),
                               metadata_json=json.dumps({**meta, **provenance,
                                                         'evidence_scope': 'sampled_ocr_frames_only',
                                                         'ocr_text_format': 'sampled_frames_v2',
                                                         'ocr_frame_parts': chunk['frames'],
                                                         'ocr_chunk_index': index,
                                                         'ocr_chunk_count': len(chunks)}, sort_keys=True)))
    return output
