"""Do not infer DCB101 yellow-indicator absence from a red-pattern description."""
import re

_MODEL = re.compile(r'(?<![A-Za-z0-9_])DCB(?:101|107|112)(?![A-Za-z0-9_])', re.IGNORECASE)
_ABSENCE = re.compile(
    r'(?:无|没有|不带|未配备)\s*黄色?(?:指示)?灯(?:参与)?|'
    r'黄(?:色)?(?:指示)?灯\s*(?:不参与|不会亮|不亮)|'
    r'\b(?:no|without)\s+(?:a\s+)?yellow\s+(?:indicator|light|LED)\b|'
    r'\byellow\s+(?:indicator|light|LED)\s+(?:does not|doesn.t|never)\s+'
    r'(?:participate|light|illuminate|activate)', re.IGNORECASE)
_UNCERTAIN = re.compile(
    r'不能|无法|不足以|未能|是否|不能据此|'
    r'\b(?:cannot|can.t|whether|not enough|not sufficient|does not establish)\b', re.IGNORECASE)


def _asserts_101_absence(text):
    for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+|'
                             r'[,，]\s*(?:但是|但|but\b|however\b)\s*', text, flags=re.IGNORECASE):
        models = list(_MODEL.finditer(sentence))
        for index, model in enumerate(models):
            if model.group().upper() != 'DCB101':
                continue
            end = models[index + 1].start() if index + 1 < len(models) else len(sentence)
            clause = sentence[model.start():end]
            for absence in _ABSENCE.finditer(clause):
                prefix = sentence[:model.start()] + clause[:absence.start()]
                # Uncertainty about an exclusion is not an assertion of it.
                if not _UNCERTAIN.search(prefix):
                    return True
    return False


def indicator_exclusion_mismatch(claim):
    if not _asserts_101_absence(claim['text']):
        return None
    manuals = [ref for ref in claim['evidence'] if ref['source_id'].startswith('M')]
    if manuals and any(not _asserts_101_absence(ref['quote']) for ref in manuals):
        return 'red_pattern_does_not_establish_yellow_indicator_absence'
    return None
