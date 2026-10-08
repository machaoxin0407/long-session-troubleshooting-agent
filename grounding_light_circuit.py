"""Bounded checks for circuit explanations added to light observations."""
import re

_CIRCUIT = r'(?:照明电路|工作灯(?:的)?(?:照明)?电路|(?:lighting|work light) circuit)'
_POWER_PART = rf'(?:{_CIRCUIT}|照明部分)'
_EVENTS = {
    'activation': re.compile(
        rf'(?:触发|激活|接通)(?:了)?\s*{_CIRCUIT}|'
        rf'\b(?:activates?|activated|triggers?|triggered|energizes?|energized)\s+'
        rf'(?:the\s+)?{_CIRCUIT}|'
        rf'{_CIRCUIT}\s+(?:is|was)\s+(?:activated|triggered)', re.IGNORECASE),
    'power': re.compile(
        rf'{_POWER_PART}(?:已|已经)?(?:通电|接通|得到供电)|'
        rf'{_CIRCUIT}\s+(?:is|was|has been)\s+(?:powered|energized)', re.IGNORECASE),
}
_CLAUSE = re.compile(r'[。！？;；，,.!\n]|\bbut\b|\bhowever\b|但是|但|然而', re.IGNORECASE)
_UNCERTAIN = re.compile(
    r'不能|无法|不足以|不代表|不意味着|未说明|未表明|未证明|不说明|不表明|不证明|'
    r'不应|并非|是否|如果|假如|不确定|没有证据|'
    r'(?:不(?:会|能)?|未(?:曾|能)?|没有)(?:触发|激活|接通)|'
    r'\b(?:cannot|can not|does not|do not|not|whether|if|unknown)\b', re.IGNORECASE)


def _asserted_events(text):
    events = set()
    for clause in _CLAUSE.split(text):
        for kind, pattern in _EVENTS.items():
            for match in pattern.finditer(clause):
                if not _UNCERTAIN.search(clause[:match.end()]):
                    events.add(kind)
    return events


def light_circuit_mismatch(claim):
    events = _asserted_events(claim['text'])
    if events and any(events - _asserted_events(ref['quote']) for ref in claim['evidence']):
        return 'light_observation_does_not_establish_circuit_mechanism'
    return None
