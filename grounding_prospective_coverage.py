"""Opt-in prospective task-coverage feedback; never edits or approves answers."""
from dataclasses import replace

from grounding_contract import (
    AnswerabilityConflictError,
    ClaimMarkupError,
    parse_revision,
)
from grounding_repair_feedback import guidance_for
from grounding_selection import expand_selection, passage_catalog
from grounding_task_completion import task_completion_rejections


def prospective_coverage_feedback(context, raw):
    selection, _ = context._split_submission(raw)
    try:
        revision = parse_revision(expand_selection(selection, passage_catalog(context.sources),
                                                  max_claims=context.max_claims),
                                  context.sources, require_assessment=True, max_claims=context.max_claims)
    except (ClaimMarkupError, AnswerabilityConflictError):
        # The existing one-call structural repair still owns this submission.
        # Do not normalize an invalid claim into prospective acceptance evidence.
        return [], ''
    if not revision.claims or revision.assessment['answerability'] == 'complete':
        return [], ''
    hypothetical = replace(revision, assessment={**revision.assessment, 'answerability': 'complete'})
    checks = task_completion_rejections(context, hypothetical)
    if not checks:
        return [], ''
    rows = [{'reason': row['reason'], 'current_submission_rejected_for_this': False,
             'scope': 'The current claims would not justify a complete assessment.',
             'guidance': guidance_for(row['reason'])} for row in checks]
    instruction = (
        '\nProspective coverage checks describe gaps that would remain if the original '
        'claims were relabelled complete. They are review instructions, not new product '
        'evidence and not a rule that the revised answer must stay partial. Re-evaluate '
        'each gap against all supplied passages and the actual revised claims. Repairing '
        'an unrelated condition does not establish a missing requested action. If evidence '
        'supports the missing part, answer it with its source; otherwise retain that '
        'specific limitation and use partial or none. Do not invent operations or discard '
        'answerable parts. Submit the original full-answer tool once for full validation. '
        'Applicable checks:\n' + '\n'.join(guidance_for(row['reason']) for row in checks))
    return rows, instruction
