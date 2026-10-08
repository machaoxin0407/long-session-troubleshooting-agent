"""Finite equivalents for a low-battery return during an unfinished cycle."""
import re


def low_battery_during_cleaning(text):
    """Bind the in-progress condition directly to returning for recharge.

    Do not borrow context from another sentence or accept an after-completion
    return. This recognizes one observed English equivalent, not entailment.
    """
    if re.search(r'[。！？\n]|(?<=[.!?])\s+|after (?:finishing|completing)', text, re.IGNORECASE):
        return False
    return bool(re.match(
        r'^\s*(?:If|When) (?:the (?:vacuum\x27s )?)?battery (?:is|gets) low '
        r'during (?:a|the) cleaning cycle, (?:the vacuum|it) returns to recharge\b',
        text, re.IGNORECASE))
