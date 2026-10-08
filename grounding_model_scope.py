"""Finite checks for model identity and unsupported whole-manual absence claims."""
import re

from grounding_indicator_exclusion import indicator_exclusion_mismatch
from grounding_model_exclusivity import model_exclusivity_mismatch

_MODEL = re.compile(r'(?<![A-Za-z0-9])(?:[A-Z]{2,8}-[A-Z]*\d[A-Z0-9-]*)(?![A-Za-z0-9])')
_LOCAL_SCOPE = re.compile(r'所提供|当前|这些|这段|本段|摘录|片段|'
                          r'\b(?:provided|supplied|selected|excerpt|extract)\b', re.IGNORECASE)
_ABSENCE = re.compile(
    r'(?:未|没有)(?:提供|列出|说明|记载|提及)|'
    r'\b(?:does not|doesn.t) (?:provide|list|mention|describe|explain)\b', re.IGNORECASE)


def model_scope_mismatch(claim):
    text = claim['text']
    exclusivity_error = model_exclusivity_mismatch(claim)
    if exclusivity_error:
        return exclusivity_error
    indicator_error = indicator_exclusion_mismatch(claim)
    if indicator_error:
        return indicator_error
    for sentence in re.split(r'[。！？\n]|(?<=[.!?])\s+', text):
        if (re.search(r'手册|\bmanual\b', sentence, re.IGNORECASE)
                and _ABSENCE.search(sentence) and not _LOCAL_SCOPE.search(sentence)):
            return 'selected_excerpt_does_not_establish_whole_manual_absence'
    # This checks explicit identifier presence, not compatibility or entailment.
    # Pure statements of evidentiary uncertainty do not assert model facts.
    models = set()
    for clause in re.split(r'[。！？;；，,\n]|(?<=[.!?])\s+', text):
        if re.search(r'(?:不能|无法|不足以|未能)(?:据此)?(?:确定|确认|证明|判断)|'
                     r'\b(?:cannot|does not) (?:establish|prove|confirm|determine)\b',
                     clause, re.IGNORECASE):
            continue
        models.update(_MODEL.findall(clause))
    for ref in claim['evidence']:
        if not ref['source_id'].startswith('M'):
            continue
        present = set(_MODEL.findall(ref['quote']))
        if models - present:
            return 'named_model_absent_from_selected_manual_passage'
    return None
