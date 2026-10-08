"""Exact numbered source blocks for repair attention, never inferred steps."""
import re


def numbered_source_blocks(catalog, passage_ids):
    records = []
    for passage_id in dict.fromkeys(passage_ids):
        passage = catalog[passage_id]
        quote = passage['quote']
        starts = list(re.finditer(r'^\s*\d+[.)]\s+', quote, re.MULTILINE))
        if len(starts) < 2:
            continue
        blocks = []
        for index, match in enumerate(starts):
            end = starts[index + 1].start() if index + 1 < len(starts) else len(quote)
            blocks.append({'start': match.start(), 'end': end,
                           'text': quote[match.start():end]})
        records.append({'passage_id': passage_id, 'source_id': passage['source_id'],
                        'blocks': blocks})
    return {'incomplete_index': True, 'independent_evidence': False,
            'applicability_verified': False, 'records': records}
