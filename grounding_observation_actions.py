"""Exclude bounded descriptive cleaning phrases from action keyword scanning.

This is not an entailment check. Only matched descriptive spans are masked;
instructions elsewhere in the same sentence remain visible to the caller.
"""
import re

_CYCLE = re.compile(
    r'\b(?:(?:after|before)\s+(?:completing|finishing)|during)\s+'
    r'(?:a|the|its|their)\s+cleaning\s+cycle\b', re.IGNORECASE)
_COMPLETION_SIGNAL = re.compile(
    r'\b(?:tones?|sounds?|signals?)\s+indicating\s+(?:successful\s+)?completion\s+of\s+'
    r'(?:a|the|its|their)\s+cleaning\s+cycle\b', re.IGNORECASE)
_DEVICE = r'(?:the |a )?(?:vacuum|robot|machine|device|appliance)'
_AUTOMATIC = re.compile(
    rf'(?:^|[.!?;,\n]|\bwhen\b|\bif\b)\s*(?:(?:when|if)\s+)?{_DEVICE}\s+(?:'
    r'(?:will\s+)?automatically\s+cleans?\b|'
    r'(?:encounters? an area of high debris concentration and\s+)?'
    r'(?:will\s+)?moves? (?:in a )?(?:forward/backward|back[- ]and[- ]forth) '
    r'motion to clean\b)', re.IGNORECASE)
_ZH_CYCLE = re.compile(r'(?:完成|结束|进行)(?:一个|本次|当前)?清洁周期(?:后|前|时|期间)')
_INDICATOR_MOTION = re.compile(
    r'\b(?:an?|the) indicator (?:illuminating|illuminates) when moving in (?:a )?'
    r'(?:forward/backward|back[- ]and[- ]forth) motion to clean\b', re.IGNORECASE)
_ZH_AUTOMATIC = re.compile(r'(?:^|[，,。；;])\s*(?:吸尘器|机器人|设备|机器)(?:会|将会)?自动清洁')


def instruction_view(text):
    """Return a scan-only view; never use this string to render an answer."""
    def mask_english(match):
        return re.sub(r'\bclean(?:s|ing)?\b', 'activity', match.group(), flags=re.IGNORECASE)

    text = _CYCLE.sub(mask_english, text)
    text = _COMPLETION_SIGNAL.sub(mask_english, text)
    text = _AUTOMATIC.sub(mask_english, text)
    text = _INDICATOR_MOTION.sub(mask_english, text)
    text = _ZH_CYCLE.sub(lambda m: m.group().replace('清洁', '作业'), text)
    return _ZH_AUTOMATIC.sub(lambda m: m.group().replace('清洁', '作业'), text)
