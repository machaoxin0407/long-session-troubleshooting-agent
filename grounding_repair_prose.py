"""Optional repair prompt view; canonical sources and submitted answers stay intact."""
import copy
import re

_IMAGE = re.compile(r'\[\[PIC:([\w.-]{1,128})\]\]')


def repair_prose_view(payload):
    """Separate renderer control tokens from source prose presented for rewriting.

    This is not sanitization of a model answer. Source selection, exact quote
    expansion, image binding and final validation still use the original context.
    Untrusted candidate claims and user questions are deliberately untouched.
    """
    result = copy.deepcopy(payload)
    changed = False

    def display(text):
        nonlocal changed
        value, count = _IMAGE.subn(' ', text)
        changed = changed or bool(count)
        return value

    for passage in result['passages'].values():
        original = passage['quote']
        projected = display(original)
        if projected != original:
            del passage['quote']
            passage['text_without_image_markers'] = projected

    for pid, lines in result.get('source_condition_lines', {}).items():
        result['source_condition_lines'][pid] = [display(line) for line in lines]
    for record in result.get('numbered_source_blocks', {}).get('records', []):
        for block in record['blocks']:
            original = block['text']
            projected = display(original)
            if projected != original:
                block['text_without_image_markers'] = projected
                del block['text']
                block['offsets_refer_to_original_quote'] = True
    if changed:
        result['source_prose_projection'] = {
            'independent_evidence': False,
            'scope': ('Only literal [[PIC:...]] rendering tokens were replaced with spaces '
                      'in source prose and attention indexes. Original sources remain '
                      'unchanged for exact quote expansion and validation. Use the existing '
                      'image_passage_ids and image_ids field for illustrations. Do not put '
                      'image identifiers or control tokens in claims.text. This display '
                      'projection does not establish support, applicability or completeness.'),
        }
    return result
