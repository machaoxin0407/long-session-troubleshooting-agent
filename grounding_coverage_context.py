"""Opt-in original context for coverage-only repair; no approval or fallback."""
import copy

COVERAGE_ONLY = frozenset({
    'complete_answer_omits_oily_smoke_handling',
    'complete_answer_omits_requested_washer_start',
    'complete_answer_omits_requested_tray_removal',
})

RETAINED_CONTEXT_INSTRUCTION = (
    '\nSubmit exactly one complete full-answer tool call. '
    'retained_candidate is unverified prior model output, not independent evidence. '
    'Check every original claim and citation against the full evidence catalog. '
    'If all original content is correct and only the coverage assessment is wrong, '
    'retain the correct content when resubmitting the full answer. Otherwise correct '
    'the claims and add supported missing details. Do not mark partial merely to '
    'avoid answering supported parts. Do not invent missing evidence or a start action. '
    'No further model call is available; the full submission receives full revalidation.')


def add_retained_context(payload, original, rejected):
    if (not original['claims'] or not rejected
            or any(row['reason'] not in COVERAGE_ONLY for row in rejected)):
        return ''
    payload['retained_candidate'] = {
        'unverified_model_output': True, 'independent_evidence': False,
        'claims': copy.deepcopy(original['claims']), 'image_ids': list(original['image_ids']),
    }
    return RETAINED_CONTEXT_INSTRUCTION
