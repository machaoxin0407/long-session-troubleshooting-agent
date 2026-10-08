"""Build evidence and bind checked claims to the final API sentence positions."""
import re
from pathlib import Path

from grounding_contract import EvidenceSource
from grounding_language import question_is_chinese
from grounding_runtime import revise_with_evidence
from response_integrity import IncompleteGenerationError


def build_evidence(*, videos, trace):
    """Build the bounded source catalog and matching citation metadata."""
    sources, metadata = [], {}
    seen = set()
    manual_budget = 16000
    for event in trace.get('events', []):
        hits = (event.get('sections', []) if event.get('kind') == 'pre_retrieval' else
                event.get('retrieval_hits', []) if event.get('kind') == 'tool_call' else [])
        for hit in hits or []:
            product = str(hit.get('product') or '')
            native_value = hit.get('parent_section_id')
            if native_value is None:
                native_value = hit.get('chunk_id', '')
            native_id = str(native_value)
            heading = str(hit.get('heading') or '')
            key = (product, native_id, heading)
            if key in seen or len(sources) >= 16 or manual_budget < 200:
                continue
            seen.add(key)
            excerpt = str(hit.get('source_text') or hit.get('text_preview') or hit.get('section_summary') or '').strip()
            if not excerpt:
                continue
            pics = list(hit.get('pics') or [])
            excerpt = f'{product} / {heading}\n{excerpt}'
            cap = min(3500, manual_budget)
            truncated = len(excerpt) > cap
            if truncated:
                excerpt = excerpt[:cap - 80].rsplit('\n', 1)[0] + '\n[Excerpt truncated; do not infer omitted steps or qualifications.]'
            manual_budget -= len(excerpt)
            source_id = f'M{len(sources) + 1}'
            sources.append(EvidenceSource(source_id, excerpt))
            metadata[source_id] = {'evidence_type': 'manual_section', 'source_id': native_id,
                                   'title': f'{product} / {heading}', 'excerpt': excerpt, 'pics': pics,
                                   'source_context': {
                                       'title': f'{product} / {heading}'[:512],
                                       'topic': heading[:512],
                                       'coverage': ('prefix_truncated' if truncated
                                                    else 'untruncated_supplied_text')}}
    video_budget = 8000
    for i, video in enumerate(videos[:8], 1):
        if video_budget < 200:
            break
        if not video.evidence_text.strip():
            continue
        source_id = f'V{i}'
        video_text = video.evidence_text[:min(2000, video_budget)]
        video_budget -= len(video_text)
        context = dict(getattr(video, 'source_context', {}))
        original_length = getattr(video, 'evidence_original_length', None)
        if original_length is None:
            original_length = len(video.evidence_text)
        if type(original_length) is not int or original_length < len(video.evidence_text):
            raise ValueError('Inconsistent original video evidence length')
        # Coverage describes supplied text, not the original video. Keep it
        # outside verbatim evidence and override caller-supplied coverage.
        context['coverage'] = ('prefix_truncated' if len(video_text) < original_length
                               else 'untruncated_supplied_text')
        source_title = context.get('title')
        # Preserve provenance in the returned citation as well as the model
        # context. A title describes the source, not verified model identity.
        citation_title = (source_title.strip()[:512] if isinstance(source_title, str) and source_title.strip()
                          else video.product_class)
        sources.append(EvidenceSource(source_id, video_text))
        metadata[source_id] = {'evidence_type': 'video_scene', 'source_id': video.scene_id,
                               'title': f'{citation_title} {video.start_seconds:.1f}-{video.end_seconds:.1f}s',
                               'excerpt': video_text, 'video': video,
                               'source_context': context}
    return sources, metadata


def grounded_response(*, api, question, draft, original_pics, videos, trace, deadline_ts):
    sources, metadata = build_evidence(videos=videos, trace=trace)
    revision = revise_with_evidence(question=question, draft=draft, sources=sources,
                                    route_name=trace.get('generation_route'), deadline_ts=deadline_ts)
    return render_revision(api=api, question=question, revision=revision,
                           sources=sources, metadata=metadata,
                           original_pics=original_pics, trace=trace)


def render_revision(*, api, question, revision, sources, metadata, original_pics, trace):
    """Render a checked revision using the exact metadata bound to its sources."""
    trace['grounding_revision'] = 'completed'
    trace['grounding_contract'] = {
        'version': 2 if revision.assessment is not None else 1, 'insufficient': revision.insufficient,
        'assessment': revision.assessment,
        'review_limited': revision.review_limited,
        'sources': [{'source_id': s.source_id, 'text': s.text} for s in sources],
        'claims': [{'text': c.text, 'evidence': [{'source_id': sid, 'quote': quote} for sid, quote in c.evidence]}
                   for c in revision.claims],
    }
    blocks, pics, references = [], [], {}
    for claim in revision.claims:
        text = claim.text
        used_ids = [sid for sid, _ in claim.evidence]
        tags = ''.join(f" [VID:{metadata[sid]['source_id']}]" for sid in used_ids
                       if metadata[sid]['evidence_type'] == 'video_scene')
        if tags:
            terminal = text[-1:] if text[-1:] in '.!?。！？' else ''
            text = text[:-1] if terminal else text
            text += tags + (terminal or '.')
        # Normalize before recording positions, and terminate each claim block.
        text = api._format_answer(text, [], 'tech')
        if text[-1:] not in '.!?。！？':
            text += '。' if re.search(r'[\u4e00-\u9fff]', text) else '.'
        blocks.append(text)
        references[len(blocks) - 1] = used_ids
        allowed = {Path(p).stem for sid in used_ids for p in metadata[sid].get('pics', [])}
        for pic in original_pics:
            if Path(pic).stem in allowed and pic not in pics:
                pics.append(pic)
                blocks.append('<PIC>')
    if revision.insufficient:
        message = revision.render(chinese=question_is_chinese(question))[0].split('\n')[-1]
        blocks.append(message)
    answer = '\n'.join(blocks)
    formatted = api._format_answer(answer, pics, 'tech')
    spans, cursor = {}, 0
    for index, block in enumerate(blocks):
        if index not in references:
            continue
        position = formatted.find(block, cursor)
        if position < 0:
            raise IncompleteGenerationError('回答格式化后无法核对证据位置。')
        spans[index] = (position, position + len(block))
        cursor = position + len(block)
    support_map = {}
    sentence_index = 0
    for match in api._ANSWER_SENTENCE_RE.finditer(formatted):
        if not match.group(0).strip():
            continue
        sentence_index += 1
        start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
        end = match.end() - (len(match.group(0)) - len(match.group(0).rstrip()))
        for index, (lo, hi) in spans.items():
            if lo <= start and end <= hi:
                for sid in references[index]:
                    support_map.setdefault(sid, []).append(f'sentence-{sentence_index}')
    citations, selected_videos = [], []
    quotes = {}
    for claim in revision.claims:
        for sid, quote in claim.evidence:
            if quote not in quotes.setdefault(sid, []):
                quotes[sid].append(quote)
    for sid, supports in support_map.items():
        meta = metadata[sid]
        citations.append(api.CitationItem(citation_id=sid, supports=supports,
                                         excerpt='\n[…]\n'.join(quotes[sid]),
                                         **{k: meta[k] for k in ('evidence_type', 'title', 'source_id')}))
        if 'video' in meta:
            selected_videos.append(meta['video'].model_copy(update={'supports': supports}))
    if any(sid not in support_map for ids in references.values() for sid in ids):
        raise IncompleteGenerationError('回答句子与证据映射不完整。')
    # Reuse media construction only; empty answer cannot issue lexical citations.
    _, manual_images = api._structured_evidence('', pics, [], trace)
    return answer, pics, selected_videos, citations, manual_images
