"""Conservative literal component-name rendering, pending runtime acceptance.

Remove only a noun gloss attached to a source-attested English identifier.
This is not translation, action repair, a semantic validator, or a fallback.
The caller must validate the returned submission with the unchanged contract.
"""
import json
import re

from grounding_contract import _strict_object, parse_revision
from grounding_selection import expand_selection, passage_catalog

_COMPONENT_HEADING = re.compile(r'\b(?:components?|parts|general description)\b|部件|组件', re.IGNORECASE)
_NUMBERED_LABEL = re.compile(r'^\s*\d{1,3}\s+([A-Za-z][A-Za-z0-9 -]{1,39})\s*$')
_CONTROL_HEADING = re.compile(r'\b(?:controls?|control panel)\b|控制面板', re.IGNORECASE)
_CONTROL_LABEL = re.compile(r'^\s*\d{1,3}\.\s+([A-Z][A-Z -]{1,39}):\s+\S.*$')
_NAMED_CONTROL_STEP = re.compile(r'^\s*\d{1,3}\.\s+Set the ([A-Z][A-Z ]{1,39} KNOB) to\b', re.MULTILINE)
# Recognize short parenthetical noun glosses, without mapping them to English.
# Component names need a numbered list and the selected passage. A printed
# control may instead be explicitly named in that passage's numbered Set step.
# Neither form licenses rewriting an unlabelled Chinese noun.
_NOUN_GLOSSES = frozenset({
    '锅', '盘', '篮', '炸篮', '篮子', '炸锅', '煎锅', '内锅', '内胆', '锅体',
    '托盘', '接油盘', '接水盘', '面板', '控制面板', '显示屏', '屏幕',
    '按钮', '开关', '旋钮', '电源线', '指示灯',
    '水位选择旋钮', '水流选择旋钮', '水选择旋钮', '洗涤选择旋钮',
    '循环选择旋钮', '洗涤定时器', '脱水定时器',
})
_PREFIX_BOUNDARIES = (
    '将', '把', '的', '从', '和', '或', '与', '取出', '拿出', '抽出', '移除',
    '拉出', '拔出', '放入', '使用', '清洁', '清洗', '触摸', '打开', '关闭',
    '拆下', '装回', '放回', '握住', '移动', '接触', '取下',
)


def _document(text):
    first = text.splitlines()[0] if text else ''
    return first.split('/', 1)[0].strip() if '/' in first else None


def _component_names(sources):
    names = {}
    for source in sources:
        lines = source.text.splitlines()
        document = _document(source.text)
        if not source.source_id.startswith('M') or not document or not lines:
            continue
        labels = []
        if _COMPONENT_HEADING.search(lines[0]):
            component_labels = [match.group(1) for line in lines[1:]
                                if (match := _NUMBERED_LABEL.fullmatch(line))]
            if len(component_labels) >= 2:
                labels.extend(component_labels)
        if _CONTROL_HEADING.search(lines[0]):
            control_labels = [match.group(1) for line in lines[1:]
                              if (match := _CONTROL_LABEL.fullmatch(line))]
            if len(control_labels) >= 2:
                labels.extend(control_labels)
                # A selector may be named with KNOB in the operating passage.
                # This only creates a lookup candidate: the complete spelling
                # must still occur literally in the claim's cited passage.
                labels.extend(label + ' KNOB' for label in control_labels if label.endswith(' SELECTOR'))
        for label in labels:
            names.setdefault((document, label.casefold()), []).append(source.source_id)
    return names


def _noun_gloss(value):
    return all(part.strip() in _NOUN_GLOSSES for part in re.split(r'[/／、]', value))


def normalize_component_glosses(context, raw):
    """Return raw JSON and an edit trail; no model call or acceptance decision."""
    selection, _ = context._split_submission(raw)
    catalog = passage_catalog(context.sources)
    # Do not use normalization to salvage malformed IDs, markup or contracts.
    parse_revision(expand_selection(selection, catalog, max_claims=context.max_claims),
                   context.sources, require_assessment=True, max_claims=context.max_claims)
    value = json.loads(raw, object_pairs_hook=_strict_object)
    names = _component_names(context.sources)
    edits = []
    for index, claim in enumerate(value['claims']):
        allowed = {}
        for pid in claim['passage_ids']:
            passage = catalog[pid]
            if not passage['source_id'].startswith('M'):
                continue
            document = _document(passage['quote'])
            if document:
                for match in _NAMED_CONTROL_STEP.finditer(passage['quote']):
                    spelling = match.group(1)
                    binding = allowed.setdefault(spelling, {'passage_ids': [], 'glossary_source_ids': []})
                    if pid not in binding['passage_ids']:
                        binding['passage_ids'].append(pid)
                    binding.setdefault('named_control_source_ids', [])
                    if passage['source_id'] not in binding['named_control_source_ids']:
                        binding['named_control_source_ids'].append(passage['source_id'])
            for (doc, term), glossary_sources in names.items():
                if doc != document:
                    continue
                # Preserve the exact surface spelling already used by the model,
                # but require its spelling to occur literally in cited evidence.
                for match in re.finditer(r'(?<![A-Za-z0-9_])' + re.escape(term)
                                         + r'(?![A-Za-z0-9_])', passage['quote'], re.IGNORECASE):
                    spelling = match.group(0)
                    binding = allowed.setdefault(spelling, {'passage_ids': [], 'glossary_source_ids': []})
                    if pid not in binding['passage_ids']:
                        binding['passage_ids'].append(pid)
                    binding['glossary_source_ids'] = sorted(set(binding['glossary_source_ids']) | set(glossary_sources))
        if not allowed:
            continue
        terms = '|'.join(re.escape(term) for term in sorted(allowed, key=len, reverse=True))
        nouns = '|'.join(re.escape(noun) for noun in sorted(_NOUN_GLOSSES, key=len, reverse=True))
        pattern = re.compile(
            r'(?<![A-Za-z0-9_])(?P<term>' + terms + r')[（(](?P<gloss>[^()（）\n]{1,30})[）)]'
            r'|(?P<before>' + nouns + r')[（(](?P<original>' + terms + r')[）)]')
        original_text = claim['text']

        def replace(match, original_text=original_text, index=index, allowed=allowed):
            term = match.group('term') or match.group('original')
            gloss = match.group('gloss') or match.group('before')
            if not _noun_gloss(gloss):
                return match.group(0)
            if match.group('before'):
                prefix = original_text[:match.start()]
                if prefix and re.search(r'[\u3400-\u9fff]$', prefix) and not prefix.endswith(_PREFIX_BOUNDARIES):
                    return match.group(0)
            edits.append({'claim_index': index, 'start': match.start(), 'end': match.end(),
                          'before': match.group(0), 'after': term, **allowed[term]})
            return term

        claim['text'] = pattern.sub(replace, original_text)
    return (json.dumps(value, ensure_ascii=False) if edits else raw), edits
