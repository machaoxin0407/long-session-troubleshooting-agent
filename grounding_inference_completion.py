"""Finite omission check for explicit single-observation inference questions.

This detects a missing delivered conclusion only. A conclusion that is present
still needs independent semantic/citation validation; presence is not support.
"""
import re

_SINGLE_OBSERVATION = re.compile(
    r'仅凭|单凭|仅靠|只凭|仅根据|\balone\b|\bby itself\b|\bon its own\b|\bbased solely\b', re.IGNORECASE)
_INFERENCE_QUESTION = re.compile(
    r'确认|证明|推断|断定|判断|\b(?:confirm|prove|establish|conclude|infer|determine|show|mean)\b', re.IGNORECASE)
_EXPLICIT_PROOF_QUESTION = re.compile(
    r'(?:能否|可否|能不能|可不可以|是否(?:足以|能够|可以))\s*(?:直接)?(?:证明|确认|推断|断定)|'
    r'(?:^|[.!?。！？]\s*)\s*(?:can|could|does|do|would)\b[^?？.!。\n]{1,180}'
    r'\b(?:prove|establish|confirm|imply)\b[^?？.!。\n]*[?？]', re.IGNORECASE)
_CONCLUSION = re.compile(
    r'确认|证明|推断|断定|判断|意味着|不能据此|不足以|不代表|'
    r'(?:不能|无法|能够|足以|可以|这|该现象|这一现象|上述观察)说明|'
    r'\b(?:confirm\w*|prov(?:e|es|ed|en)|establish\w*|conclud\w*|infer\w*|'
    r'determin\w*|demonstrat\w*|impl(?:y|ies)|mean(?:s)?|sufficient|enough)\b|'
    r'^(?:是的|不是|不能|可以|不可以|yes\b|no\b)', re.IGNORECASE)


def inference_completion_rejections(context, revision):
    inference_question = (_SINGLE_OBSERVATION.search(context.question)
                          and _INFERENCE_QUESTION.search(context.question)) or (
                              _EXPLICIT_PROOF_QUESTION.search(context.question))
    if not revision.claims or not inference_question:
        return []
    # Assessment prose and missing_parts are never delivered answer claims.
    if any(_CONCLUSION.search(claim.text.strip()) for claim in revision.claims):
        return []
    return [{'reason': 'inference_question_conclusion_missing'}]
