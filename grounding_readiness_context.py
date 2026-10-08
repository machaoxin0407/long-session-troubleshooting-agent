"""Detect loss of explicit procedure context in observed readiness claims.

This finite exclusion does not establish entailment for claims that pass it.
"""
import re


def readiness_context_mismatch(text, quote):
    if not quote.startswith('Espresso Machine /'):
        return None
    if '/ Reset to Factory Settings /' in quote:
        context = r'恢复出厂|出厂重置|复位|factory reset|reset(?:ting)?(?: to factory settings)?'
    elif ('/ First Use or After a Long Period of Non-Use /' in quote
          and 'Press the Lungo button to rinse the machine. Repeat 3 times.' in quote):
        context = r'首次|初次|第一次|长时间未使用|first use|initial (?:setup|use)|long.{0,20}non.use|rinse|rinsing|冲洗'
    else:
        return None
    # Only observed light/readiness claims: unrelated statements are left alone.
    if not re.search(r'常亮|灯光稳定|稳定的灯光|steady lights?|lights? (?:are |remain )?steady', text, re.IGNORECASE):
        return None
    if not re.search(r'就绪|准备好|准备就绪|\bready\b', text, re.IGNORECASE):
        return None
    # Explicit epistemic limitations are not positive readiness assertions.
    sentences = re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text)
    positive = [s for s in sentences if re.search(r'就绪|准备好|准备就绪|\bready\b', s, re.IGNORECASE)
                and not re.search(r'不代表|不能|无法|does not|do not|cannot|not ready', s, re.IGNORECASE)]
    if not positive:
        return None
    # A context can head the claim, but a negated mention cannot supply it.
    for match in re.finditer(context, text, re.IGNORECASE):
        if not re.search(r'不是|并非|无需|不需要|\bnot\b|without', text[max(0, match.start() - 18):match.start()], re.IGNORECASE):
            return None
    return 'readiness_procedure_context_missing'
