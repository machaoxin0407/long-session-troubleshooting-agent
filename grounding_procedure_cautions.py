"""Finite, source-bound procedure cautions; passing is not semantic approval."""
import re

KEYBOARD_CAUTION = '注意工具请勿刮擦上盖面板'
KEYBOARD_REASON = 'switch_removal_panel_caution_missing'


def switch_removal_source(quote):
    return (quote.startswith('功能键盘手册 /')
            and '/ 轴体拆卸方法\n' in quote
            and KEYBOARD_CAUTION in quote)


def _delivers_removal(text):
    # Require the actual tool/clip/removal combination, not a product name or
    # an observation. A pure refusal or quotation of a missing procedure is
    # not an instruction to which we should append a new operation.
    for sentence in re.split(r'[。！？\n]|(?<=[.!?])\s+', text):
        if re.search(r'不能|无法|未说明|未提供|不(?:能|要|应|得)|禁止|不要|'
                     r'cannot|does not (?:describe|provide)|do not|must not|never',
                     sentence, re.IGNORECASE):
            continue
        if (re.search(r'拔轴器|\bswitch puller\b', text, re.IGNORECASE)
                and re.search(r'按下.{0,10}卡扣|\bpress.{0,25}\bclips?\b', sentence, re.IGNORECASE)
                and re.search(r'拆卸|拆轴|取下轴体|\bremov\w*\b', sentence, re.IGNORECASE)):
            return True
    return False


def _retains_caution(text):
    for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
        # Conditional or metalinguistic mention is not an unconditional caution.
        if re.search(r'如果|仅当|只有|(?:不必|无需|不用)(?:注意|避免|遵守|担心|防止)|未说明|未要求|没有(?:要求|说)|'
                     r'\bif\b|\bunless\b|need not|not necessary|does not (?:say|require)',
                     sentence, re.IGNORECASE):
            continue
        if re.search(r'(?:请勿|切勿|不要|不得|避免|不可|勿)(?:用工具|让工具|使工具)?'
                     r'(?:刮擦|刮伤|划伤)(?:上盖面板|上盖的面板)', sentence):
            return True
        if re.search(r"\b(?:do not|don't|must not|never)\s+scratch\s+(?:the\s+)?top(?:\s+cover)?\s+panel\b|"
                     r'\bavoid\s+scratching\s+(?:the\s+)?top(?:\s+cover)?\s+panel\b',
                     sentence, re.IGNORECASE):
            return True
    return False


def procedure_caution_rejections(revision):
    operating_sources = {
        sid for claim in revision.claims if _delivers_removal(claim.text)
        for sid, quote in claim.evidence if sid.startswith('M') and switch_removal_source(quote)
    }
    if not operating_sources:
        return []
    retained = any(
        _retains_caution(claim.text)
        and any(sid in operating_sources and switch_removal_source(quote) for sid, quote in claim.evidence)
        for claim in revision.claims
    )
    return [] if retained else [{'reason': KEYBOARD_REASON}]
