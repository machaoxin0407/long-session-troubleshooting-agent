"""Finite source-relation checks, not a general factual correctness score."""
import re

_ALIAS = re.compile(
    r'(?<![A-Za-z0-9_])SDHC\s*(?:卡|cards?)?\s*[（(]?\s*(?:也称为|又称|即|就是|等同于|'
    r'(?:is |are )?(?:also )?(?:called|known as))\s*TF\s*(?:卡|cards?)?', re.IGNORECASE)
_NEGATED_ALIAS = re.compile(r'不能|无法|不应|不是|不等同|未证明|\b(?:not|cannot|whether)\b', re.IGNORECASE)
_WEAK_WEEK = re.compile(
    r'(?:建议|推荐|可以考虑)[^。！？;；\n]{0,30}(?:至少)?每周[^。！？;；\n]{0,20}(?:一次|1次)|'
    r'\b(?:recommend(?:ed)?|advisable|optional)[^.!?;\n]{0,65}\b(?:once a week|weekly)\b',
    re.IGNORECASE)


def _anaphoric_seal_frequency(text, match):
    """Resolve explicit 'this operation' only to the adjacent seal action."""
    before = text[:match.start()]
    previous = re.search(
        r'(?P<action>[^。！？;；.!?\n]+)[。.][\s]*(?:It is\s+)?$',
        before, re.IGNORECASE)
    if not previous:
        return False
    action = previous['action']
    if not re.search(
            r'(?:清洁|清洗|冲洗)(?:橡胶)?密封圈(?:周围)?\s*$|'
            r'\bclean\s+around\s+(?:the\s+)?(?:rubber\s+)?seal\s*$',
            action, re.IGNORECASE):
        return False
    # Include the remainder of this sentence, since Chinese puts 此操作
    # after 一次. A bare frequency or a named new operation is not anaphora.
    sentence = re.split(r'[。！？;；.!?\n]', text[match.start():])[0]
    return bool(re.fullmatch(
        r'(?:建议|推荐|可以考虑)(?:至少)?每周(?:进行|执行)?(?:一次|1次)'
        r'(?:此|该|上述)(?:清洁|清洗|冲洗)?操作|'
        r'recommended\s+to\s+(?:do|perform|repeat)\s+(?:this|this operation)\s+'
        r'(?:at least\s+)?(?:once a week|weekly)', sentence.strip(), re.IGNORECASE))


def _asserts_alias(text):
    for clause in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
        for match in _ALIAS.finditer(clause):
            if not _NEGATED_ALIAS.search(clause[:match.end()]):
                return True
    return False


def source_relation_mismatch(claim):
    text = claim['text']
    # Co-occurrence or an ambiguous parenthesis after a list does not
    # identify a specific member as an alias. Check every selected quote.
    if (_asserts_alias(text)
            and any(not _asserts_alias(ref['quote']) for ref in claim['evidence'])):
        return 'storage_alias_not_established_by_each_citation'
    for match in _WEAK_WEEK.finditer(text):
        prefix = re.split(r'[。！？;；\n]', text[:match.start()])[-1]
        if re.search(r'不能|不是|并非|不应|\bnot\b', prefix, re.IGNORECASE):
            continue
        explicit_cleaning = re.search(r'清洁|清洗|冲洗|clean|rinse', match.group(), re.IGNORECASE)
        # A bare trailing frequency can qualify the immediately preceding seal
        # cleaning action. Do not transfer it across sentences or to a following
        # inspection/storage action just because cleaning appears elsewhere.
        trailing_seal_frequency = (
            re.fullmatch(r'(?:建议|推荐|可以考虑)(?:至少)?每周(?:一次|1次)', match.group())
            and re.search(r'(?:清洁|清洗|冲洗)(?:橡胶)?密封圈(?:周围)?[，,\s]*$', prefix)
            and (match.end() == len(text) or text[match.end()] in '。！？;；\n，,'))
        if not explicit_cleaning and not trailing_seal_frequency and not _anaphoric_seal_frequency(text, match):
            continue
        for ref in claim['evidence']:
            source = ref['quote']
            # This finite case concerns the supplied cleaning instruction,
            # not unrelated weekly recommendations elsewhere in a document.
            if (re.search(r'Gently clean around the rubber seal\.\s*At least once a week\.',
                          source, re.IGNORECASE)
                    and re.search(r'清洁|清洗|冲洗|clean|rinse', text, re.IGNORECASE)):
                return 'cleaning_minimum_frequency_weakened_to_recommendation'
    return None
