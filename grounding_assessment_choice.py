"""Diagnostic-only choice of reassessment or full replacement in one call.

No API path imports this experiment. Retention is not semantic approval: the
assembled submission must pass the complete existing validator again.
"""
import copy
import json

from grounding_contract import GroundingContractError, _strict_object
from grounding_final_repair import repair_rejections
from grounding_final_tool import NAME

ASSESSMENT_NAME = 'reassess_retained_answer'
_COVERAGE_ONLY = frozenset({
    'complete_answer_omits_oily_smoke_handling',
    'complete_answer_omits_requested_washer_start',
    'complete_answer_omits_requested_tray_removal',
})


def _eligible(context, raw):
    value = json.loads(raw, object_pairs_hook=_strict_object)
    errors = repair_rejections(context, raw)
    if (not value['claims'] or not errors
            or any(row['reason'] not in _COVERAGE_ONLY for row in errors)):
        raise GroundingContractError('Assessment choice requires coverage-only rejections')
    return value


def assessment_choice_request(request, context, original_raw):
    original = _eligible(context, original_raw)
    result = copy.deepcopy(request)
    if (len(result['tools']) != 1
            or result['tools'][0]['function']['name'] != NAME
            or result.get('tool_choice') != 'required'):
        raise GroundingContractError('Assessment choice requires one original tool and required selection')
    payload = json.loads(result['messages'][1]['content'])
    if (payload['question'] != context.question
            or payload['passages'] != context.evidence_payload()['passages']
            or 'retained_candidate' in payload):
        raise GroundingContractError('Assessment choice context mismatch')
    payload['retained_candidate'] = {
        'unverified_model_output': True, 'independent_evidence': False,
        'claims': original['claims'], 'image_ids': original['image_ids'],
    }
    schema = result['tools'][0]['function']['parameters']
    result['tools'].append({'type': 'function', 'function': {
        'name': ASSESSMENT_NAME,
        'description': 'Revise only answerability and limitations after checking every retained '
                       'claim against evidence. Choose the original full-answer tool if any claim, '
                       'citation, image or missing supported content must change.',
        'parameters': {'type': 'object', 'properties': {
            'assessment': copy.deepcopy(schema['properties']['assessment']),
            'insufficient': copy.deepcopy(schema['properties']['insufficient']),
        }, 'required': ['assessment', 'insufficient'], 'additionalProperties': False},
    }})
    result['messages'][1]['content'] = json.dumps(payload, ensure_ascii=False)
    result['messages'][0]['content'] += (
        '\nFor this diagnostic, submit exactly one of the two tools. '
        'The original full-answer tool remains available to correct content or add '
        'supported missing details. Use reassess_retained_answer only if every retained '
        'claim and citation is correct and only the coverage assessment needs correction. '
        'Retained model text is not independent evidence. Do not mark partial merely to '
        'avoid answering supported parts. Do not invent missing evidence or a start action. '
        'No further model call is available; either choice receives full revalidation.')
    return result


def retained_context_request(request, context, original_raw):
    """Ablation: the same retained content, but only the original full tool."""
    choice = assessment_choice_request(request, context, original_raw)
    result = copy.deepcopy(request)
    result['messages'][1] = choice['messages'][1]
    result['messages'][0]['content'] += (
        '\nFor this diagnostic, submit exactly one complete full-answer tool call. '
        'retained_candidate is unverified prior model output, not independent evidence. '
        'Check every original claim and citation against the full evidence catalog. '
        'If all original content is correct and only the coverage assessment is wrong, '
        'retain the correct content when resubmitting the full answer. Otherwise correct '
        'the claims and add supported missing details. Do not mark partial merely to '
        'avoid answering supported parts. Do not invent missing evidence or a start action. '
        'No further model call is available; the full submission receives full revalidation.')
    return result


def validate_assessment_choice(context, original_raw, tool_name, raw, *, deadline_ts):
    original = _eligible(context, original_raw)
    if tool_name == ASSESSMENT_NAME:
        value = json.loads(raw, object_pairs_hook=_strict_object)
        if not isinstance(value, dict) or set(value) != {'assessment', 'insufficient'}:
            raise GroundingContractError('Unexpected reassessment fields')
        candidate = copy.deepcopy(original)
        candidate.update(value)
        assembled = json.dumps(candidate, ensure_ascii=False)
    elif tool_name == NAME:
        assembled = raw
    else:
        raise GroundingContractError('Unexpected assessment-choice tool')
    revision = context.validate(assembled, tool_names=[NAME],
                                finish_reason='tool_calls', deadline_ts=deadline_ts)
    if repair_rejections(context, assembled):
        raise GroundingContractError('Assessment choice failed full revalidation')
    return assembled, revision
