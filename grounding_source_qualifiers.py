"""Diagnostic lexical index; exact offsets never establish semantic support."""
import re

_QUALIFIER = re.compile(
    r'\b(?:up|down|toward|towards|above|below|before|after|only|without|no|not|never|'
    r'unless|if|in\s+case)\b|\d+(?:[.,]\d+)?|'
    r'向下|向上|之前|之后|不得|不要|仅|如果|无', re.IGNORECASE)


def source_qualifier_spans(catalog):
    """Index matches with their complete source line, without inferred relations."""
    records = []
    for passage_id, passage in catalog.items():
        source = passage['quote']
        offset = 0
        for full_line in source.splitlines(keepends=True):
            line = full_line.rstrip('\r\n')
            matches = list(_QUALIFIER.finditer(line))
            if matches:
                records.append({
                    'passage_id': passage_id, 'source_id': passage['source_id'],
                    'line': {'start': offset, 'end': offset + len(line), 'text': line},
                    'spans': [{'start': offset + m.start(), 'end': offset + m.end(),
                               'text': m.group()} for m in matches],
                })
            offset += len(full_line)
    return {'incomplete_lexical_index': True, 'independent_evidence': False,
            'untrusted_source_data': True, 'records': records}
