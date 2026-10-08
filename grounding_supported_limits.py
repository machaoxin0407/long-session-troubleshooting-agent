"""Finite source-bound empty-answer checks; never generate replacement prose."""
import re

BATTERY_LIMIT_REASON = 'supported_battery_service_limit_missing_from_empty_answer'


def battery_service_limit_source(text):
    """Require the supplied replacement section and its explicit service limit."""
    return (
        text.startswith('Media Player /')
        and '/ Replacing the Battery\n' in text
        and 'please contact manufacturer and DO NOT intend to do so by yourself.' in text
        and 'This process only can be done by manufacturer’s technician.' in text
    )


def supported_limit_rejections(context, revision):
    # Nonempty claims still need all existing relevance, support and omission
    # checks. This rule does not certify their content or trust assessment notes.
    if revision.claims:
        return []
    question = context.question
    product = re.search(r'\bmedia\s+player\b|\b(?:e[ -]?)?reader\b|阅读器', question, re.IGNORECASE)
    battery = re.search(r'\bbatter(?:y|ies)\b|电池', question, re.IGNORECASE)
    replacement = re.search(r'\breplac(?:e|ing|ement)\b|更换|换电池', question, re.IGNORECASE)
    if not (product and battery and replacement):
        return []
    if any(s.source_id.startswith('M') and battery_service_limit_source(s.text)
           for s in context.sources):
        return [{'reason': BATTERY_LIMIT_REASON}]
    return []
