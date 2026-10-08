"""Finite same-language modality checks; absence of a match is not entailment."""
import re

_OPTIONAL = re.compile(
    r'(?:无需|无须|不必|不需要)(?P<zh>[^。！？；;，,\n]{2,100})|'
    r'\b(?:do(?:es)? not need to|need not|not necessary to|no need to)\s+'
    r'(?P<en>[^.!?;,\n]{3,160})', re.IGNORECASE)
_FORBIDDEN = re.compile(
    r'(?:不允许|禁止|不得|切勿)(?P<zh>[^。！？；;，,\n]{2,160})|'
    r'\b(?:must not|do not|never|not allowed to|forbidden to)\s+'
    r'(?P<en>[^.!?;,\n]{3,200})', re.IGNORECASE)


def _actions(text, pattern):
    for clause in re.split(r'[。！？；;，,\n]|(?<=[.!?])\s+', text):
        for match in pattern.finditer(clause):
            prefix = clause[:match.start()]
            # Meta-language and quoted contrasts are not asserted directives.
            if re.search(r'["“”‘’]|并非|不是|不代表|不等于|不能理解|不能将|不能把|'
                         r'\b(?:not mean|does not say|does not state|not imply|'
                         r'not the same|rather than)\b', prefix, re.IGNORECASE):
                continue
            language = 'zh' if match.group('zh') else 'en'
            action = match.group(language).strip().rstrip(' .。')
            # Do not strip conditions and create an unconditional prohibition.
            if re.search(r'如果|除非|仅当|之前|之后|时|\b(?:if|unless|when|before|after)\b',
                         action, re.IGNORECASE):
                continue
            if language == 'en':
                action = re.sub(r'\s+', ' ', action).casefold()
            yield language, action


def unnecessary_as_prohibited(claim):
    """Reject explicit strengthened wording tied to the same verb phrase.

    No fuzzy matching, model names, question IDs or bilingual guesses. A separate
    prohibition on the same action in this passage is allowed. Every selected
    passage is checked independently; a second citation cannot repair the first.
    """
    forbidden = list(_actions(claim['text'], _FORBIDDEN))
    if not forbidden:
        return None
    for item in claim['evidence']:
        quote = item['quote']
        explicit = list(_actions(quote, _FORBIDDEN))
        for language, action in _actions(quote, _OPTIONAL):
            for target_language, target in forbidden:
                if language != target_language:
                    continue
                boundary = '' if language == 'zh' else r'(?=$|\W)'
                if not re.match(re.escape(action) + boundary, target):
                    continue
                if any(lang == language and re.match(re.escape(action) + boundary, text)
                       for lang, text in explicit):
                    continue
                return 'unnecessary_action_misrepresented_as_prohibited'
    return None
