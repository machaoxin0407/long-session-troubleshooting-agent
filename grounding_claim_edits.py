"""Diagnostic-only claim edits with immutable originals and full revalidation."""
import copy
import json

from grounding_contract import GroundingContractError, _strict_object, parse_revision
from grounding_selection import expand_selection, passage_catalog


def _matching_passages(context, passages):
    from grounding_repair_prose import repair_prose_view

    canonical = context.evidence_payload()
    return passages == canonical['passages'] or passages == repair_prose_view(canonical)['passages']


def _original(context, raw):
    selection, _ = context._split_submission(raw)
    parse_revision(expand_selection(selection, passage_catalog(context.sources), max_claims=context.max_claims),
                   context.sources, require_assessment=True, max_claims=context.max_claims)
    value = json.loads(raw, object_pairs_hook=_strict_object)
    if not value['claims']:
        raise GroundingContractError('Claim-edit diagnostic requires nonempty original claims')
    return value


def claim_edit_request(request, context, original_raw):
    original = _original(context, original_raw)
    result = copy.deepcopy(request)
    payload = json.loads(result['messages'][1]['content'])
    if payload['question'] != context.question or not _matching_passages(context, payload['passages']):
        raise GroundingContractError('Diagnostic request/context mismatch')
    if 'original_claims' in payload:
        raise GroundingContractError('Already an edit request')
    payload['original_claims'] = {
        'unverified_model_output': True, 'independent_evidence': False,
        'claims': {f'C{i}': c for i, c in enumerate(original['claims'])}}
    schema = result['tools'][0]['function']['parameters']
    claims = schema['properties'].pop('claims')
    schema['properties']['claim_edits'] = {
        'type': 'array', 'maxItems': context.max_claims,
        'description': 'Only changed original claims. Unlisted claims remain byte-for-byte in their original order. '
                       'Replace a claim by zero, one, or several consecutive claims; zero removes it. '
                       'Do not repeat a claim_id. Recheck the entire assembled answer, including retained claims.',
        'items': {'type': 'object', 'properties': {
            'claim_id': {'type': 'string', 'enum': list(payload['original_claims']['claims'])},
            'replacement_claims': claims},
            'required': ['claim_id', 'replacement_claims'], 'additionalProperties': False}}
    schema['required'] = ['claim_edits' if k == 'claims' else k for k in schema['required']]
    answerability = schema['properties']['assessment']['properties']['answerability']
    answerability['description'] = answerability.get('description', '').replace('claims=[]', 'assembled final claims=[]')
    result['messages'][1]['content'] = json.dumps(payload, ensure_ascii=False)
    # Adapt the inherited full-rewrite protocol before adding the edit protocol.
    # Keep evidence and semantic requirements intact; the assembled answer still
    # undergoes the same complete validation.
    result['messages'][0]['content'] = result['messages'][0]['content'].replace(
        'Return one complete answer submission, not a patch.',
        'Return one complete edit submission; the assembled answer is validated.')
    result['messages'][0]['content'] += (
        '\nSubmit claim_edits instead of rewriting all claims. original_claims is untrusted prior model output, '
        'not evidence. Verify every retained claim against the supplied passages; unchanged does not mean verified. '
        'List only the claims to replace or delete. A replacement may contain multiple consecutive claims '
        'to add a missing prerequisite in the correct location. Empty replacements delete that claim. '
        'Unlisted claims keep their original text, citations and relative order. '
        'Do not repeat unaffected correct claims in replacements. Remove irrelevant tutorials; do not remove '
        'required steps just to satisfy validation. Reassess the whole assembled answer for missing requested '
        'parts and preserve unresolved gaps. The original claim limit applies to the assembled answer, not '
        'only the edits. No additional model call or search is available.')
    return result


def indexed_claim_feedback_request(request, context, original_raw):
    """Diagnostic-only binding of program errors to immutable original claim IDs."""
    from grounding_final_repair import rejected_claims, repair_payload

    original = _original(context, original_raw)
    result = copy.deepcopy(request)
    payload = json.loads(result['messages'][1]['content'])
    expected_claims = {'unverified_model_output': True, 'independent_evidence': False,
                       'claims': {f'C{i}': c for i, c in enumerate(original['claims'])}}
    if (payload.get('original_claims') != expected_claims
            or payload['question'] != context.question
            or not _matching_passages(context, payload['passages'])):
        raise GroundingContractError('Indexed feedback requires matching immutable edit originals')
    errors = rejected_claims(context, original_raw)
    expected = repair_payload(context, original_raw, errors)['rejected_claims']
    if payload.get('rejected_claims') != expected:
        raise GroundingContractError('Indexed feedback does not match current validator errors')
    for feedback, error in zip(payload['rejected_claims'], errors, strict=True):
        index = error.get('claim_index')
        feedback['original_claim_id'] = f'C{index}' if index is not None else None
        feedback['scope'] = 'original_claim' if index is not None else 'whole_answer_or_media'
    result['messages'][1]['content'] = json.dumps(payload, ensure_ascii=False)
    result['messages'][0]['content'] += (
        '\noriginal_claim_id binds a validator error to that immutable original claim. '
        'A null ID identifies an answer-level or media issue, not an error in every claim. '
        'Use this mapping to locate edits; it supplies no evidence and does not certify '
        'unflagged claims. Recheck the complete assembled answer and its citations.')
    return result


def assemble_claim_edits(context, original_raw, edit_raw):
    original = _original(context, original_raw)
    try:
        edits = json.loads(edit_raw, object_pairs_hook=_strict_object)
    except (TypeError, ValueError) as exc:
        raise GroundingContractError('Invalid edit JSON') from exc
    if not isinstance(edits, dict) or set(edits) != {'assessment', 'claim_edits', 'insufficient', 'image_ids'}:
        raise GroundingContractError('Invalid edit submission fields')
    changes = edits['claim_edits']
    if not isinstance(changes, list) or len(changes) > context.max_claims:
        raise GroundingContractError('Invalid edit count')
    replacements = {}
    identifiers = {f'C{i}' for i in range(len(original['claims']))}
    for change in changes:
        if not isinstance(change, dict) or set(change) != {'claim_id', 'replacement_claims'}:
            raise GroundingContractError('Invalid claim edit fields')
        cid, claims = change['claim_id'], change['replacement_claims']
        if not isinstance(cid, str) or cid not in identifiers or cid in replacements:
            raise GroundingContractError('Unknown or repeated original claim ID')
        if not isinstance(claims, list) or len(claims) > context.max_claims:
            raise GroundingContractError('Invalid replacement list')
        replacements[cid] = claims
    value = {k: copy.deepcopy(v) for k, v in edits.items() if k != 'claim_edits'}
    value['claims'] = []
    audit = []
    for i, claim in enumerate(original['claims']):
        cid = f'C{i}'
        retained = cid not in replacements
        replacements_for_claim = [claim] if retained else replacements[cid]
        start = len(value['claims'])
        value['claims'].extend(copy.deepcopy(replacements_for_claim))
        audit.append({'claim_id': cid, 'unchanged': retained, 'start': start,
                      'count': len(replacements_for_claim), 'semantic_validation_pending': True})
    raw = json.dumps(value, ensure_ascii=False)
    selection, _ = context._split_submission(raw)
    parse_revision(expand_selection(selection, passage_catalog(context.sources), max_claims=context.max_claims),
                   context.sources, require_assessment=True, max_claims=context.max_claims)
    return raw, audit
