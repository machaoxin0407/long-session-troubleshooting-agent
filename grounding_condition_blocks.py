"""Keep explicit forward references with their contiguous numbered source list."""
import re

_FORWARD = re.compile(r'\b(?:below|following(?:\s+(?:conditions|cases|situations))?)\s*[:：]?\s*$'
                      r'|(?:以下|下列)(?:情况|情形|条件)?\s*[:：]?\s*$', re.IGNORECASE)
_NUMBER = re.compile(r'^\s*(\d+)[.)、]\s*\S')


def condition_list_block(lines, index):
    """Return an exact span and next index; no inference about applicability."""
    if not _FORWARD.search(lines[index].rstrip('\r\n')):
        return None
    cursor = index + 1
    expected = 1
    end = cursor
    while cursor < len(lines):
        line = lines[cursor]
        if not line.strip():
            cursor += 1
            continue
        number = _NUMBER.match(line)
        if number:
            if int(number.group(1)) != expected:
                break
            expected += 1
        elif expected > 1 and line[:1].isspace():
            # Only indented continuations inherit an item; another unindented
            # heading or paragraph belongs to the full catalog, not this list.
            pass
        else:
            break
        end = cursor + 1
        cursor += 1
    if expected == 1:
        return None
    return ''.join(lines[index:end]).rstrip('\r\n'), end
