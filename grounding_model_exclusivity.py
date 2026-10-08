"""Reject explicit model exclusivity when selected evidence never asserts it.

This finite negative check does not establish entailment when a passage does
contain exclusivity language; the remaining evidence checks still apply.
"""
import re

_ID = r'(?<![A-Za-z0-9_])([A-Z]{1,8}-?\d[A-Z0-9-]*)(?![A-Za-z0-9_])'
_PATTERNS = [
    re.compile(_ID + r'\s*(?:型号)?(?:所)?(?:独有|特有|专有)', re.IGNORECASE),
    re.compile(r'(?:仅有|只有|仅)\s*' + _ID + r'\s*(?:型号)?(?:具备|具有|支持|会|提供|发出|显示)', re.IGNORECASE),
    re.compile(r'\b(?:unique|exclusive)\s+to\s+(?:the\s+)?' + _ID, re.IGNORECASE),
    re.compile(_ID + r"(?:'s|’s)\s+(?:unique|exclusive)\b", re.IGNORECASE),
    re.compile(r'\bonly\s+(?:the\s+)?' + _ID + r'\s+(?:has|supports|provides|shows|emits|can)\b', re.IGNORECASE),
]
_UNCERTAIN = re.compile(
    r'不能|无法|不足以|未能|尚未|是否|并非|不是|不一定|'
    r'\b(?:cannot|can.t|whether|not|never|doesn.t)\b', re.IGNORECASE)


def _asserted_models(text):
    models = set()
    for clause in re.split(r'[。！？;；\n]|(?<=[.!?])\s+|'
                           r'[,，]\s*(?:但是|但|but\b|however\b)\s*', text,
                           flags=re.IGNORECASE):
        for pattern in _PATTERNS:
            for match in pattern.finditer(clause):
                if not _UNCERTAIN.search(clause[:match.start()]):
                    models.add(match.group(1).upper())
    return models


def model_exclusivity_mismatch(claim):
    asserted = _asserted_models(claim['text'])
    if not asserted:
        return None
    refs = claim['evidence']
    # Absence of a corresponding assertion is sufficient to reject this
    # particular unsupported strengthening, never to invent its replacement.
    if any(asserted - _asserted_models(ref['quote']) for ref in refs):
        return 'described_model_behavior_does_not_establish_exclusivity'
    return None
