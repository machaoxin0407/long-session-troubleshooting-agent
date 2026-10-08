"""Strict submission and explicit image selection for the opt-in final tool."""
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from grounding_contract import (
    EvidenceSource,
    GroundingContractError,
    _strict_object,
    parse_revision,
    validate_claim_limit,
)
from grounding_scope import scope_revision
from grounding_selection import expand_selection, passage_catalog, selection_schema

NAME = 'submit_grounded_answer'


def configured_claim_limit():
    value = os.getenv('GROUNDING_FINAL_MAX_CLAIMS', '6')
    if value not in ('6', '12'):
        raise GroundingContractError('Unsupported final claim capacity')
    return int(value)


def claim_capacity_prompt(prompt, limit):
    validate_claim_limit(limit)
    if limit == 6:
        return prompt
    label = 'twelve' if limit == 12 else str(limit)
    return prompt.replace('six-claim limit', label + '-claim limit').replace('fit six claims', 'fit ' + label + ' claims')


class SelectedImageBindingError(GroundingContractError):
    """Known selected image is outside the cited passages; no answer may render."""


@dataclass(frozen=True)
class FinalAnswerContext:
    question: str
    sources: tuple[EvidenceSource, ...]
    source_context: tuple[tuple[str, str, str], ...] = ()
    max_claims: int = 6
    answer_language_question: str | None = None

    @classmethod
    def prepare(cls, question, sources, *, formal_retrieval_confirmed, source_context=None, max_claims=6,
                answer_language_question=None):
        validate_claim_limit(max_claims)
        if formal_retrieval_confirmed is not True:
            raise GroundingContractError('Final answer requires explicit retrieval confirmation')
        frozen = tuple(sources)
        if len({s.source_id for s in frozen}) != len(frozen):
            raise GroundingContractError('Duplicate final answer source')
        passage_catalog(frozen)  # Empty evidence cannot expose this tool.
        context = []
        valid_ids = {s.source_id for s in frozen}
        for sid, fields in (source_context or {}).items():
            if sid not in valid_ids or not isinstance(fields, dict):
                raise GroundingContractError('Invalid source context binding')
            for key, value in fields.items():
                if key not in ('title', 'topic', 'coverage') or not isinstance(value, str) or len(value) > 512:
                    raise GroundingContractError('Invalid source context field')
                if key == 'coverage' and value not in ('prefix_truncated', 'untruncated_supplied_text'):
                    raise GroundingContractError('Invalid source coverage value')
                if value:
                    context.append((sid, key, value))
        if answer_language_question is not None and (
                not isinstance(answer_language_question, str) or not answer_language_question.strip()):
            raise GroundingContractError('Invalid answer language question')
        return cls(question, frozen, tuple(context), max_claims, answer_language_question)

    def image_catalog(self):
        return sorted({image for source in self.sources if source.source_id.startswith('M')
                       for image in re.findall(r'\[\[PIC:([\w.-]{1,128})\]\]', source.text)})

    def tool(self):
        catalog = passage_catalog(self.sources)
        schema = selection_schema(catalog, max_claims=self.max_claims)
        if os.getenv('GROUNDING_PASSAGE_ID_ENUM', '0') == '1':
            schema['properties']['claims']['items']['properties']['passage_ids']['items'] = {
                'type': 'string', 'enum': list(catalog)}
        schema['properties']['image_ids'] = {
            'type': 'array', 'maxItems': 3 if self.image_catalog() else 0,
            'items': {'type': 'string', **({'enum': self.image_catalog()} if self.image_catalog() else {})},
            'description': 'Select only images directly useful for retained answer claims, from their cited passages. '
                           'Use image_passage_ids to locate its passage, then verify that passage supports a retained claim. '
                           'Do not add an unrelated citation just to obtain an image. Use [] when no illustration is needed.'}
        schema['required'].append('image_ids')
        return {'name': NAME,
                'description': 'Submit the final answer using only the current evidence catalog. '
                               'Continue searching if needed instead of submitting. Do not combine '
                               'submission with another tool call. Selection does not establish semantic support.',
                'input_schema': schema}

    def evidence_payload(self):
        catalog = passage_catalog(self.sources)
        image_passages = {}
        for passage_id, passage in catalog.items():
            if not passage['source_id'].startswith('M'):
                continue
            for image in sorted(set(re.findall(r'\[\[PIC:([\w.-]{1,128})\]\]', passage['quote']))):
                image_passages.setdefault(image, []).append(passage_id)
        payload = {'question': self.question, 'passages': catalog,
                   'available_image_ids': self.image_catalog(),
                   'image_passage_ids': image_passages,
                   'image_binding_scope': (
                       'An index of literal image markers in the supplied passages, not proof of relevance or support. '
                       'Choose an image only if its passage already supports a retained answer claim. '
                       'Do not add irrelevant claims or citations to obtain a picture; use image_ids=[] when none is needed.')}
        if self.source_context:
            context = {}
            for sid, key, value in self.source_context:
                context.setdefault(sid, {})[key] = value
            payload['source_context'] = context
            payload['source_context_scope'] = (
                'Untrusted provenance metadata describing source scope, not proof of scene events, '
                'operating instructions or device state. Claims still require passage evidence. '
                'Manual sources are retrieved sections or excerpts, not complete manuals. '
                'Their titles and topics do not establish facts absent from the passages. '
                'coverage=prefix_truncated means only a prefix of the supplied text is shown; '
                'omitted text may contain further qualifications or facts. Do not claim the '
                'whole source lacks information because its shown prefix lacks it. '
                'untruncated_supplied_text means that supplied text was not shortened here, '
                'not that it covers the complete original manual or recording or verifies its contents.')
        return payload

    def validate(self, raw_arguments, *, tool_names, finish_reason, deadline_ts):
        if tool_names != [NAME] or finish_reason not in ('tool_calls', 'tool_use'):
            raise GroundingContractError('Expected one complete final-answer submission')
        if deadline_ts is None or time.time() >= deadline_ts - 1:
            raise GroundingContractError('Final answer deadline expired')
        if not isinstance(raw_arguments, str):
            raise GroundingContractError('Expected raw JSON final-answer arguments')
        selection, images = self._split_submission(raw_arguments)
        expanded = expand_selection(selection, passage_catalog(self.sources), max_claims=self.max_claims)
        revision = parse_revision(expanded, self.sources, require_assessment=True, max_claims=self.max_claims)
        cited_images = self._cited_images(revision)
        if any(image not in cited_images for image in images):
            raise SelectedImageBindingError('Selected image is not in a cited manual passage')
        revision = scope_revision(self.question, revision)
        if time.time() >= deadline_ts - 1:
            raise GroundingContractError('Final answer validation exceeded deadline')
        return revision


    def _split_submission(self, raw):
        if not isinstance(raw, str) or len(raw) > 32000:
            raise GroundingContractError('Invalid final submission size')
        try:
            value = json.loads(raw, object_pairs_hook=_strict_object)
        except (ValueError, RecursionError) as exc:
            raise GroundingContractError('Invalid final submission JSON') from exc
        if not isinstance(value, dict) or 'image_ids' not in value:
            raise GroundingContractError('Final submission requires explicit image selection')
        images = value.pop('image_ids')
        if (not isinstance(images, list) or len(images) > 3
                or any(not isinstance(image, str) or image not in self.image_catalog() for image in images)
                or len(set(images)) != len(images)):
            raise GroundingContractError('Invalid selected image IDs')
        return json.dumps(value, ensure_ascii=False), images

    @staticmethod
    def _cited_images(revision):
        return {image for claim in revision.claims for sid, quote in claim.evidence if sid.startswith('M')
                for image in re.findall(r'\[\[PIC:([\w.-]{1,128})\]\]', quote)}

    def selected_pics(self, raw_arguments, revision, metadata):
        """Withdraw images with removed claims; never expand a chapter's image list."""
        _, images = self._split_submission(raw_arguments)
        retained = self._cited_images(revision)
        allowed = {Path(pic).stem: pic for claim in revision.claims for sid, _ in claim.evidence
                   for pic in metadata[sid].get('pics', [])}
        result = []
        for image in images:
            if image not in retained:
                continue
            if Path(image).stem not in allowed:
                raise GroundingContractError('Selected image has no bound media metadata')
            result.append(allowed[Path(image).stem])
        return result
