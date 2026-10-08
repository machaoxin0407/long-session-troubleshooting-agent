"""Bounded exclusions for inferring component health from light observations."""
import re

from grounding_light_circuit import light_circuit_mismatch
from grounding_light_independence import light_independence_mismatch

_COMPONENT = r'(?:电路|扳机(?:开关)?|开关|触发功能|circuit|trigger(?: switch)?|switch)'
_COMPONENT_START = re.compile(_COMPONENT, re.IGNORECASE)
_HEALTH = re.compile(
    rf'{_COMPONENT}[^。！？;；,.!\n]{{0,45}}?(?:功能正常|正常|无故障|'
    r'\b(?:working|functioning|operating) (?:normally|properly|correctly)\b|'
    r'\bfault[- ]free\b)', re.IGNORECASE)
_DENIAL = re.compile(
    r'不(?:表明|说明|证明|证实)|'
    r'不能|无法|不足以|不代表|不意味着|并非|是否|不正常|未(?:能)?(?:证明|确认|说明|表明|证实)|'
    r'\b(?:cannot|can not|does not|do not|not|whether)\b', re.IGNORECASE)
_LIGHT = re.compile(r'工作灯|指示灯|work light|indicator light|\bLED\b', re.IGNORECASE)
_CLAUSE = re.compile(r'[。！？;；，,.!\n]|\bbut\b|\bhowever\b|但是|但|然而', re.IGNORECASE)
_LIGHT_RESPONSE_SUBJECT = re.compile(
    r'(?:时|后)\s*(?:工作灯|指示灯)(?:会|将|保持)?(?:点亮|亮起|亮着|亮)|'
    r'\b(?:is pressed|is pulled)\s+(?:the\s+)?(?:work light|indicator light|LED)\s+'
    r'(?:is|turns|lights|remains)\b', re.IGNORECASE)
_FAULT = re.compile(
    rf'(?P<component>{_COMPONENT})(?:的)?(?:功能|工作|运行)?'
    r'(?:不正常|异常|(?:出现|存在|发生|有)?故障|(?:已经|已)?损坏|失灵)|'
    r'(?P<english>circuit|trigger(?: switch)?|switch)\s+'
    r'(?:(?:is|was|appears|seems)\s+(?:to be\s+)?(?:faulty|broken|defective)|'
    r'(?:is|was)\s+not\s+(?:working|functioning|operating)\s+(?:normally|properly|correctly))',
    re.IGNORECASE)
_FAULT_UNCERTAINTY = re.compile(
    r'不(?:表明|说明|证明|证实)|'
    r'不能|无法|不足以|不代表|不意味着|并非|是否|如果|假如|若|未(?:能)?(?:证明|确认|说明|表明|证实)|'
    r'\b(?:cannot|can not|does not|do not|not|whether|if|unknown)\b', re.IGNORECASE)


def _asserted_fault_components(text):
    components = set()
    for clause in _CLAUSE.split(text):
        for match in _FAULT.finditer(clause):
            # Inspect only the prefix: "is not working" asserts a fault, while
            # "does not show that ... is not working" denies that inference.
            if _FAULT_UNCERTAINTY.search(clause[:match.start()]):
                continue
            component = (match['component'] or match['english']).lower()
            component = ('circuit' if component in ('电路', 'circuit') else
                         'trigger' if component.startswith(('扳机', 'trigger')) else 'switch')
            components.add(component)
    return components


def _positive_component_health(text):
    for clause in _CLAUSE.split(text):
        for component in _COMPONENT_START.finditer(clause):
            match = _HEALTH.match(clause, component.start())
            if not match:
                continue
            # The component may be an antecedent action, not the subject of
            # "normal": pressing a trigger can be followed by a light response.
            # Check each component start so a later circuit-health assertion is
            # still inspected even when an earlier trigger match is excluded.
            if _LIGHT_RESPONSE_SUBJECT.search(match.group()):
                continue
            if not _DENIAL.search(clause[:match.end()]):
                return True
    return False


def component_health_errors(claim):
    """Collect all detected component errors for the single repair opportunity."""
    errors = []
    circuit = light_circuit_mismatch(claim)
    if circuit:
        errors.append(circuit)
    independence = light_independence_mismatch(claim)
    if independence:
        errors.append(independence)
    healthy = _positive_component_health(claim['text'])
    faults = _asserted_fault_components(claim['text'])
    for ref in claim['evidence']:
        quote = ref['quote']
        if not _LIGHT.search(quote):
            continue
        if healthy and not _positive_component_health(quote):
            errors.append('light_observation_does_not_establish_component_health')
        if faults - _asserted_fault_components(quote):
            errors.append('light_observation_does_not_establish_component_fault')
    return list(dict.fromkeys(errors))


def component_health_mismatch(claim):
    """Preserve the original single-reason API and rejection priority."""
    errors = component_health_errors(claim)
    return errors[0] if errors else None
