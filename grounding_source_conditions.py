"""Conservative indexes of explicit parenthetical source conditions.

No model/device facts are added. Every field points into the original passage;
this incomplete index is diagnostic data, never independent claim evidence.
"""
import re

_LINE = re.compile(
    r'^(?:\s*(?:\d+[.)]|[•*-])\s*)?(?P<action>[^()\n]+?)\s*'
    r'\((?P<marker>in case)\s+(?P<body>[^()\n]+)\)'
    r'\s*(?:\[\[PIC:[\w.-]+\]\]\s*)*$', re.IGNORECASE)
_ALTERNATIVE = re.compile(r"(?P<condition>[^\"']+?)\s+(?P<value>[\"'][^\"'\n]+[\"'])$")
_SETTING = re.compile(r"Set\s+[^\"']+?\s+to\s+(?P<value>[\"'][^\"'\n]+[\"'])\.?$", re.IGNORECASE)
_CONDITION = re.compile(r'[A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,7}$')


def source_condition_structure(catalog):
    result = []
    for passage_id, passage in catalog.items():
        source = passage['quote']
        offset = 0
        for full_line in source.splitlines(keepends=True):
            line = full_line.rstrip('\r\n')
            match = _LINE.fullmatch(line)
            if match:
                body = match.group('body')
                alternative = _ALTERNATIVE.fullmatch(body)
                setting = _SETTING.fullmatch(match.group('action'))
                condition = alternative.group('condition') if alternative else body
                if _CONDITION.fullmatch(condition) and (not alternative or setting):
                    def span(start, end, offset=offset, source=source):
                        return {'start': offset+start, 'end': offset+end,
                                'text': source[offset+start:offset+end]}
                    row = {'passage_id': passage_id, 'source_id': passage['source_id'],
                           'line': span(0, len(line)),
                           'kind': 'default_with_conditional_alternative' if alternative else 'conditional_action',
                           'action': span(*match.span('action')),
                           'condition': span(match.start('marker'), match.start('body')+len(condition))}
                    if alternative:
                        row['default_value'] = span(match.start('action')+setting.start('value'),
                                                    match.start('action')+setting.end('value'))
                        row['alternative_value'] = span(match.start('body')+alternative.start('value'),
                                                        match.start('body')+alternative.end('value'))
                    result.append(row)
            offset += len(full_line)
    return {'incomplete_index': True, 'independent_evidence': False, 'records': result}
