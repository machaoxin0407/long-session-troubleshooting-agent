"""Conservative request-wide context limits; never a provider tokenizer claim."""
from __future__ import annotations

import copy
import json
import os
import re
from contextvars import ContextVar

from session_context import fit_text
from session_memory import _terms

PROTECTED_TEXTS = ContextVar('protected_context_texts', default=())
OPTIONAL_TEXTS = ContextVar('optional_context_texts', default=None)
BUDGET_EVENTS = ContextVar('context_budget_events', default=None)


class ContextBudgetExceeded(ValueError):
    def __init__(self):
        super().__init__('context budget exceeded; shorten the question or attachments')


def optional_text(text, priority):
    parts = OPTIONAL_TEXTS.get()
    if parts is not None and text:
        parts.append((priority, text))


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, 'model_dump'):
        return _plain(value.model_dump(exclude_none=True))
    if hasattr(value, '__dict__'):
        return _plain(vars(value))
    return value


def _estimate(value, image_reserve):
    images = 0

    def scrub(item):
        nonlocal images
        if isinstance(item, dict):
            if item.get('type') in {'image', 'image_url', 'input_image'}:
                images += 1
                # Encoded pixels are not language tokens. Keep their metadata in
                # the estimate and reserve visual tokens separately, without
                # modifying the actual request or fetching a remote image.
                return {'type': item['type'], 'visual_payload': '[preserved]'}
            return {k: scrub(v) for k, v in item.items()}
        if isinstance(item, list):
            return [scrub(v) for v in item]
        return item

    size = len(json.dumps(scrub(_plain(value)), ensure_ascii=False).encode('utf-8'))
    return size + images * image_reserve, images


def _content_text(message):
    content = message.get('content', '')
    if isinstance(content, str):
        return content
    return '\n'.join(b.get('text', '') for b in content if isinstance(b, dict)) if isinstance(content, list) else ''


def _has_image(value):
    if isinstance(value, dict):
        return value.get('type') in {'image', 'image_url', 'input_image'} or any(_has_image(v) for v in value.values())
    return isinstance(value, list) and any(_has_image(v) for v in value)


def _compact_retrieval(text, target, protected):
    """Shorten known manual result blocks, retaining every source header."""
    starts = list(re.finditer(r'(?m)^\[(?:SECTION_FULL|\d+)\] [^\n]+', text))
    if not starts:
        return fit_text(text, target), 0, 0
    prefix = text[:starts[0].start()]
    blocks = [text[match.start():starts[n + 1].start() if n + 1 < len(starts) else len(text)]
              for n, match in enumerate(starts)]
    queries, seen, duplicates, omitted = _terms('\n'.join(protected)), {}, 0, 0
    for n, block in enumerate(blocks):
        header, _, content = block.partition('\n')
        key = content.strip()
        if key and key in seen:
            blocks[n] = header + '\n[重复检索内容已省略；同文来源: ' + seen[key] + ']\n'
            duplicates += 1
        elif key:
            seen[key] = header
    order = sorted(range(len(blocks)), key=lambda n: (len(queries & _terms(blocks[n])), -n))
    for n in order:
        if len((prefix + ''.join(blocks)).encode('utf-8')) <= target:
            break
        header = blocks[n].partition('\n')[0]
        blocks[n] = header + '\n[此来源正文因请求预算省略，可按来源回查]\n'
        omitted += 1
    return prefix + ''.join(blocks), duplicates, omitted


def _remove_old_exchange(messages, protected):
    cycles = []
    for i, message in enumerate(messages):
        if message.get('role') != 'assistant':
            continue
        ids = {t['id'] for t in message.get('tool_calls', [])}
        content = message.get('content')
        if isinstance(content, list):
            ids |= {b['id'] for b in content if b.get('type') == 'tool_use'}
        if ids:
            cycles.append((i, ids))
    for i, ids in cycles[:-1]:
        if any(p in _content_text(messages[i]) for p in protected):
            continue
        matched, touched = set(), []
        for j in range(i + 1, len(messages)):
            m = messages[j]
            if m.get('role') == 'assistant':
                break
            if m.get('role') == 'tool' and m.get('tool_call_id') in ids:
                matched.add(m['tool_call_id'])
                touched.append((j, None))
            elif isinstance(m.get('content'), list):
                hits = {b.get('tool_use_id') for b in m['content'] if b.get('type') == 'tool_result'} & ids
                if hits:
                    matched |= hits
                    touched.append((j, [b for b in m['content'] if b.get('tool_use_id') not in ids]))
        if matched != ids:
            continue
        for j, remainder in reversed(touched):
            if remainder:
                messages[j]['content'] = remainder
            else:
                del messages[j]
        del messages[i]
        return True
    return False


def prepare_body(body, *, trim=True):
    """Return a copied wire body within the estimated input+output budget.

    System instructions, tool schemas, current user text and images are fixed.
    Tool results retain their call IDs and a labelled excerpt when shortened.
    """
    result = copy.deepcopy(_plain(body))
    window = int(os.getenv('LLM_CONTEXT_WINDOW_TOKENS', '32768'))
    safety = int(os.getenv('LLM_CONTEXT_SAFETY_TOKENS', '1024'))
    visual = int(os.getenv('LLM_IMAGE_TOKEN_RESERVE', '8192'))
    output = int(result.get('max_tokens', result.get('max_completion_tokens', 0)))
    if min(window, visual) <= 0 or min(safety, output) < 0:
        raise ValueError('invalid context budget settings')
    messages = result.get('messages', [])
    protected = PROTECTED_TEXTS.get()
    if not protected:
        users = [m for m in messages if m.get('role') == 'user' and _content_text(m)]
        protected = tuple(dict.fromkeys((_content_text(users[0]), _content_text(users[-1])))) if users else ()
    before, images = _estimate(result, visual)
    removed_cycles = removed_parts = trimmed_results = 0
    duplicate_results = omitted_results = 0

    def over():
        return _estimate(result, visual)[0] + output + safety > window

    if trim:
        while over() and _remove_old_exchange(messages, protected):
            removed_cycles += 1
        candidates = []
        for m in messages:
            if m.get('role') == 'tool' and isinstance(m.get('content'), str):
                candidates.append((m, 'content'))
            if isinstance(m.get('content'), list):
                candidates.extend((b, 'content') for b in m['content']
                                  if b.get('type') == 'tool_result' and isinstance(b.get('content'), str))
        for holder, key in sorted(candidates, key=lambda x: -len(x[0][x[1]])):
            if not over():
                break
            text = holder[key]
            if any(p in text for p in protected) or len(text.encode()) <= 512:
                continue
            excess = _estimate(result, visual)[0] + output + safety - window
            target = max(256, len(text.encode()) - excess - 128)
            holder[key], duplicates, omitted = _compact_retrieval(text, target, protected)
            duplicate_results += duplicates
            omitted_results += omitted
            trimmed_results += 1
        # Remove only explicitly registered optional fragments. The protected
        # current question is supplied independently by the API worker.
        for _, fragment in sorted(OPTIONAL_TEXTS.get() or (), key=lambda x: -x[0]):
            if not over():
                break
            if any(fragment in p or p in fragment for p in protected):
                continue
            changed = False
            for m in messages:
                if m.get('role') == 'system':
                    continue
                content = m.get('content')
                if isinstance(content, str) and fragment in content:
                    m['content'] = content.replace(fragment, '[历史内容因请求预算省略]')
                    changed = True
                elif isinstance(content, list):
                    for b in content:
                        if b.get('type') == 'text' and fragment in b.get('text', ''):
                            b['text'] = b['text'].replace(fragment, '[历史内容因请求预算省略]')
                            changed = True
            removed_parts += changed
        # Standalone historical messages may be supplied by internal callers.
        for i in reversed(range(len(messages))):
            if not over():
                break
            m = messages[i]
            if (m.get('role') not in {'user', 'assistant'} or not isinstance(m.get('content'), str)
                    or any(p in m['content'] for p in protected) or m.get('tool_calls') or _has_image(m)):
                continue
            del messages[i]
            removed_parts += 1
    after, _ = _estimate(result, visual)
    event = {'counter': 'utf8_byte_estimate', 'window': window, 'before': before,
             'input_estimate': after, 'output_reserved': output, 'safety_reserved': safety,
             'images': images, 'image_reserved_each': visual, 'removed_tool_exchanges': removed_cycles,
             'removed_optional_parts': removed_parts, 'trimmed_tool_results': trimmed_results,
             'duplicate_retrieval_blocks': duplicate_results, 'omitted_retrieval_blocks': omitted_results,
             'within_budget': after + output + safety <= window,
             'reason': 'context_budget_exceeded' if after + output + safety > window else
                       'context_compacted' if after < before else 'unchanged'}
    events = BUDGET_EVENTS.get()
    if events is not None:
        events.append(event)
    if not event['within_budget']:
        raise ContextBudgetExceeded()
    return result
