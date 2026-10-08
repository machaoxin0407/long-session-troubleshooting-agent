"""Finite exclusion for asserted light/mechanism independence, not entailment."""
import re

_LIGHT = r'(?:工作灯|指示灯|work light|indicator light|LED)'
_PART = r'(?:电机|马达|夹头|motor|chuck)'
_SPAN = r'[^。！？;；，,.!\n]{0,60}?'
_RELATION = re.compile(
    rf'{_LIGHT}{_SPAN}(?:独立于|不依赖于?|与{_SPAN}(?:相互独立|互不影响|无关))'
    rf'{_SPAN}|{_LIGHT}{_SPAN}\b(?:independent of|independently of|'
    rf'does not depend on|unrelated to)\b{_SPAN}{_PART}'
    rf'|{_LIGHT}{_SPAN}{_PART}{_SPAN}(?:相互独立|互不影响|彼此独立)'
    rf'|{_PART}{_SPAN}{_LIGHT}{_SPAN}(?:相互独立|互不影响|彼此独立)',
    re.IGNORECASE)
_PARTS = re.compile(_PART, re.IGNORECASE)
_CLAUSE = re.compile(r'[。！？;；，,.!\n]|\bbut\b|\bhowever\b|但是|但|然而', re.IGNORECASE)
_UNCERTAIN = re.compile(
    r'无法|不能|不足以|不代表|不意味着|并非|是否|如果|假如|未说明|未表明|未证明|'
    r'不说明|不表明|不证明|不确定|'
    r'\b(?:cannot|can not|does not show|does not prove|not independent|'
    r'not known|unknown|whether|if)\b', re.IGNORECASE)


def _asserted_parts(text):
    parts = set()
    for clause in _CLAUSE.split(text):
        if not _RELATION.search(clause) or _UNCERTAIN.search(clause):
            continue
        for match in _PARTS.finditer(clause):
            parts.add('chuck' if match.group().lower() in ('夹头', 'chuck') else 'motor')
    return parts


def light_independence_mismatch(claim):
    asserted = _asserted_parts(claim['text'])
    if asserted and any(asserted - _asserted_parts(ref['quote']) for ref in claim['evidence']):
        return 'light_observation_does_not_establish_mechanical_independence'
    return None
