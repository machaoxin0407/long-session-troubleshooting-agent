"""Finite exclusions for purchasing obligations and cited service details."""
import re

_PURCHASE = re.compile(
    r'(?:需要|必须|需)[^。！？;；，,\n]{0,12}(?:购买|买)|'
    r'\b(?:must|need to|needs to|have to|required to)\s+(?:buy|purchase)\b|'
    r'(?:^|[.!?\n])\s*(?:Buy|Purchase)\s+', re.IGNORECASE)
_CONTACT = re.compile(r'联系|\bcontact\b', re.IGNORECASE)
_METHODS = {'phone': r'电话|\bphone\b|\btelephone\b',
            'email': r'电子邮件|邮件|\be-?mail\b'}
_TECHNICIAN = re.compile(r"厂商(?:的)?技术员|制造商(?:的)?技术员|manufacturer(?:'s|’s)? technician", re.IGNORECASE)
_CLAUSES = re.compile(r'[。！？;；\n]|(?<=[.!?])\s+')
_UNCERTAIN = re.compile(r'不能(?:据此)?(?:确定|确认|证明|推断)|无法(?:确定|确认|证明)|'
                        r'\b(?:cannot|does not) (?:establish|prove|confirm|infer|conclude)\b', re.IGNORECASE)
_REPAIR_ACTION = (
    r'(?:(?:disassemble|dissemble|dismantle)'
    r'(?:\s+(?:it|(?:the|this)\s+(?:device|reader)))?\s+(?:or|and)\s+)?repair\b')
_REPAIR_BAN = re.compile(
    rf'\b(?:do not|must not|should not|never)\s+(?:(?:attempt|intend)\s+to\s+)?{_REPAIR_ACTION}|'
    r'\b(?:warns?|instructs?|tells?|advises?)\s+(?:users?|you)\s+not\s+to\s+'
    rf'(?:(?:attempt|intend)\s+to\s+)?{_REPAIR_ACTION}|'
    r'\b(?:warned|instructed|told|advised)\s+not\s+to\s+'
    rf'(?:(?:attempt|intend)\s+to\s+)?{_REPAIR_ACTION}|'
    r'(?:不要|不得|禁止|不可)(?:尝试|自行|擅自){0,2}'
    r'(?:(?:拆卸|拆解)(?:设备|阅读器)?(?:或|和|及))?(?:维修|修理)', re.IGNORECASE)
_FAULT_SCOPE = re.compile(
    r'\b(?:if|when|in case)\b[^。！？;；.!?\n]{0,180}'
    r'\b(?:problem|damage|damaged|malfunction|overheat\w*|spill|wet|fault|work)\b|'
    r'(?:如果|若|当)[^。！？;；\n]{0,100}(?:故障|损坏|过热|进水|液体|问题|工作)', re.IGNORECASE)
_SERVICE_STATES = {
    'wet': r'\bwet\b|进水|淋湿',
    'overheat': r'\boverheat\w*\b|过热',
    'fall_damage': r'\b(?:damag\w*[^.!?;\n]{0,25}fall\w*|fall\w*[^.!?;\n]{0,25}damag\w*)\b|摔坏|跌落[^。；\n]{0,8}损坏',
    'liquid_ingress': r'\b(?:spill\w*|liquid[^.!?;\n]{0,30}(?:fall\w*|enter\w*))\b|液体[^。；\n]{0,8}(?:进入|落入)',
    'failed_correct_use': r'\bcorrect\s+operation[^.!?;\n]{0,90}(?:still|cannot|can not)[^.!?;\n]{0,35}work\b|'
                          r'正确操作[^。；\n]{0,15}仍[^。；\n]{0,10}(?:无法|不能)正常工作',
}
_BROAD_FAULT = re.compile(r'\b(?:problems?|issues?|faults?|malfunctions?)\b|问题|故障', re.IGNORECASE)


def _preserves_listed_service_scope(clause, quote):
    # This finite guard recognizes the enumerated states in this source family.
    # Merely saying "if there is a problem" does not preserve an enumerated scope.
    if not _FAULT_SCOPE.search(clause) or _BROAD_FAULT.search(clause):
        return False
    if re.search(r'\b(?:if|when)\b[^,;.!?]{0,35}\bnot\b[^,;.!?]{0,25}\b(?:wet|overheat\w*|damaged)\b|'
                 r'(?:如果|若|当)[^，。；]{0,15}(?:未|没有|不)(?:进水|过热|损坏)', clause, re.IGNORECASE):
        return False
    stated = {name for name, pattern in _SERVICE_STATES.items() if re.search(pattern, clause, re.IGNORECASE)}
    supported = {name for name, pattern in _SERVICE_STATES.items() if re.search(pattern, quote, re.IGNORECASE)}
    return bool(stated) and stated <= supported


def _unqualified_conditional_repair_ban(text, quote):
    # A finite source pattern, not a general entailment checker. This manual's
    # "shall ... encounter the below" qualifies the repair restriction itself.
    # A condition in a later sentence must not qualify an earlier prohibition.
    source_bans = [clause for clause in _CLAUSES.split(quote) if _REPAIR_BAN.search(clause)]
    if not source_bans or not all(re.search(
            r'\bshall\s+your\s+device\s+encounter\s+the\s+below\b', clause,
            re.IGNORECASE) for clause in source_bans):
        return False
    return any(_REPAIR_BAN.search(clause) and not _UNCERTAIN.search(clause)
               and not _preserves_listed_service_scope(clause, quote) for clause in _CLAUSES.split(text))


def _requires_purchase(text):
    for match in _PURCHASE.finditer(text):
        prefix = re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text[:match.start()])[-1]
        if not re.search(r'不代表|不意味着|不能|并非|不是|无需|不\s*$|'
                         r'\b(?:does not mean|cannot infer|not necessary|no need|whether)\b|\bnot\s*$',
                         prefix, re.IGNORECASE):
            return True
    return False


def purchase_service_mismatch(claim):
    text = claim['text']
    for ref in claim['evidence']:
        quote = ref['quote']
        if _unqualified_conditional_repair_ban(text, quote):
            return 'conditional_service_restriction_broadened'
        if (_requires_purchase(text)
                and re.search(r'not included|不随|未随|不包含|不附带', quote, re.IGNORECASE)
                and not _requires_purchase(quote)):
            return 'not_included_does_not_require_purchase'
    for clause in _CLAUSES.split(text):
        if _UNCERTAIN.search(clause):
            continue
        if _CONTACT.search(clause):
            methods = {name for name, pattern in _METHODS.items() if re.search(pattern, clause, re.IGNORECASE)}
            for ref in claim['evidence']:
                if any(not re.search(_METHODS[name], ref['quote'], re.IGNORECASE) for name in methods):
                    return 'service_contact_method_missing_from_citation'
        if (_TECHNICIAN.search(clause)
                and any(not _TECHNICIAN.search(ref['quote']) for ref in claim['evidence'])):
            return 'service_technician_actor_missing_from_citation'
    return None


def service_citation_gaps(claim):
    """List every absent contact/actor detail, without certifying other text.

    The rejection function intentionally returns one reason. Repair diagnostics
    need all per-reference absences to avoid exchanging one wrong citation for
    another when a compound claim mixes facts from different passages.
    """
    required = set()
    for clause in _CLAUSES.split(claim['text']):
        if _UNCERTAIN.search(clause):
            continue
        if _CONTACT.search(clause):
            required.update(name for name, pattern in _METHODS.items()
                            if re.search(pattern, clause, re.IGNORECASE))
        if _TECHNICIAN.search(clause):
            required.add('manufacturer_technician')
    patterns = {**_METHODS, 'manufacturer_technician': _TECHNICIAN.pattern}
    return [{'evidence_index': i, 'missing_details': sorted(
        name for name in required if not re.search(patterns[name], ref['quote'], re.IGNORECASE))}
        for i, ref in enumerate(claim['evidence'])
        if any(not re.search(patterns[name], ref['quote'], re.IGNORECASE) for name in required)]
