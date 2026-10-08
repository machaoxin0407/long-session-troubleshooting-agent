"""Token-budgeted, same-session context with recoverable original messages.

No tokenizer dependency or download: UTF-8 bytes are a conservative estimate for
common byte-based tokenizers, not an exact count for an arbitrary provider model.
"""
from __future__ import annotations

import hashlib
import json
import re
import zlib

from request_cancellation import check_request_cancelled
from session_memory import _terms, excerpt, observe

FIELDS = {
    "corrections": "用户更正",
    "constraints": "否定条件与约束",
    "device_models": "设备型号",
    "confirmed_symptoms": "用户确认的现象",
    "attempted_actions": "已尝试操作",
    "open_questions": "待解决问题",
    "unverified_assistant_advice": "旧客服建议（未核验）",
}


def estimate_tokens(text: str) -> int:
    return len(text.encode("utf-8"))


def fit_text(text: str, budget: int, *, extract: bool = False) -> str:
    if estimate_tokens(text) <= budget:
        return text
    # At most three UTF-8 bytes per ordinary CJK character; measure afterwards
    # because astral Unicode needs four. Always retain an explicit omission marker.
    compress = observe if extract else excerpt
    limit = min(len(text) - 1, budget)
    while limit >= 32:
        candidate = compress(text, limit)
        size = estimate_tokens(candidate)
        if size <= budget:
            return candidate
        limit = min(limit - 1, max(31, int(limit * budget / size) - 8))
    return "[内容已移出上下文，可按轮次回查]" if budget >= 60 else "[省略]"


def render_recall(selected, question: str, budget: int) -> str:
    query = _terms(question)
    selected = selected[:max(1, min(3, budget // 300))]
    allowance = budget // len(selected)
    lines = []
    for _, turn_id, _created_at, turn in sorted(selected, key=lambda item: item[1]):
        def relevant(text, available):
            sentences = [s for s in re.split(r"(?<=[。！？])|(?<=[.!?])\s+|\n+", text) if s]
            matches = sorted(sentences, key=lambda s: len(query & _terms(s)), reverse=True)
            if matches and query & _terms(matches[0]):
                return fit_text(matches[0], available)
            return fit_text(text, available, extract=True)

        user_label = f"[turn={turn_id}; user] "
        answer_label = f"[turn={turn_id}; assistant未核验] "
        user = relevant(turn['question'], max(32, allowance - 80))
        line = user_label + user
        remaining = allowance - estimate_tokens(line + '\n' + answer_label)
        if remaining >= 48:
            line += '\n' + answer_label + relevant(turn['answer'], remaining)
        if estimate_tokens('\n'.join(lines + [line])) <= budget:
            lines.append(line)
    return '\n'.join(lines)


def _size(messages: list[dict]) -> int:
    return estimate_tokens(json.dumps(messages, ensure_ascii=False))


def _messages(turns: list[dict], budget: int) -> list[dict]:
    messages = []
    for turn in turns:
        for role, key in (("user", "question"), ("assistant", "answer")):
            messages.append({"role": role, "content": turn[key],
                             "source": f" [turn={turn['id']}; unix={turn['created_at']:.3f}]",
                             "turn_id": turn['id']})
    if _size(messages) <= budget:
        return messages
    # Large recent turns are offloaded; the canonical database row remains intact.
    allowance = max(60, budget // max(1, len(messages)) - 150)
    messages = [{**m, "content": fit_text(m['content'], allowance),
                 "offloaded": estimate_tokens(m['content']) > allowance} for m in messages]
    while messages and _size(messages) > budget:
        # Drop complete oldest pairs, never an orphaned tool/message fragment.
        messages = messages[2:]
    return messages


def _sources(turns: list[dict], budget: int) -> list[dict]:
    """Bound a single model input; no recursive summaries of summaries."""
    result = []
    # Include recent older records first when the canonical archive is too large.
    for turn in reversed(turns):
        candidate = result + [turn]
        if estimate_tokens(json.dumps(candidate, ensure_ascii=False)) <= budget:
            result.append(turn)
        elif not result:
            allowance = max(64, (budget - 256) // 2)
            while allowance >= 64:
                clipped = {**turn, "question": fit_text(turn['question'], allowance, extract=True),
                           "answer": fit_text(turn['answer'], allowance, extract=True)}
                if estimate_tokens(json.dumps([clipped], ensure_ascii=False)) <= budget:
                    result.append(clipped)
                    break
                allowance //= 2
    result.sort(key=lambda t: t['id'])
    return result


def validate_summary(value, sources: list[dict], budget: int) -> str:
    """Require actual source turns, role-correct quotations and bounded output.

    Exact quotes validate provenance, not entailment of the model's paraphrase.
    Every model observation remains explicitly unverified in the answering prompt.
    """
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        raise ValueError("invalid summary fields")
    raw = {turn['id']: turn for turn in sources}
    lines = []
    for field, label in FIELDS.items():
        items = value[field]
        if not isinstance(items, list) or len(items) > 20:
            raise ValueError("invalid summary list")
        for item in items:
            if not isinstance(item, dict) or set(item) != {"text", "sources"}:
                raise ValueError("invalid observation")
            if not isinstance(item['text'], str) or not item['text'].strip():
                raise ValueError("empty observation")
            citations = item['sources']
            if not isinstance(citations, list) or not 1 <= len(citations) <= 5:
                raise ValueError("missing source")
            rendered = []
            for citation in citations:
                if not isinstance(citation, dict) or set(citation) != {"turn_id", "role", "quote"}:
                    raise ValueError("invalid citation")
                turn_id, role, quote = citation['turn_id'], citation['role'], citation['quote']
                expected_role = 'assistant' if field == 'unverified_assistant_advice' else 'user'
                if type(turn_id) is not int or turn_id not in raw or role != expected_role:
                    raise ValueError("invalid source role or turn")
                key = 'question' if role == 'user' else 'answer'
                if (not isinstance(quote, str) or not quote.strip()
                        or '[中间内容已省略]' in quote or quote not in raw[turn_id][key]):
                    raise ValueError("quote absent from source")
                rendered.append(f"[turn={turn_id}; {role}] 原话：{quote}")
            lines.append(f"{label}: {item['text']} {' '.join(rendered)}")
    if not lines:
        raise ValueError("empty summary")
    header = "结构化模型摘要（未重新核验；当前更正优先，旧建议须依本轮证据核验）：\n"
    selected = []
    # Preserve entire entries, including provenance; never clip their citations.
    for line in lines:
        if estimate_tokens(header + '\n'.join(selected + [line])) <= budget:
            selected.append(line)
    if not selected:
        raise ValueError("summary exceeds budget")
    return header + '\n'.join(selected)


def _fallback(turns: list[dict], budget: int) -> str:
    header = "规则回退摘录（摘要不可用或超预算；旧回答未重新核验）：\n"
    lines = []
    # Preserve newer user conditions before unverified older assistant prose.
    for role, key in (("user", "question"), ("assistant", "answer")):
        for turn in reversed(turns):
            remaining = budget - estimate_tokens(header + '\n'.join(lines)) - 80
            if remaining < 96:
                break
            text = fit_text(turn[key], min(remaining, max(128, budget // 4)), extract=True)
            line = f"[turn={turn['id']}; {role}] {text}"
            if estimate_tokens(header + '\n'.join(lines + [line])) <= budget:
                lines.append(line)
    return header + '\n'.join(lines)


def build_context(store, session_id: str, *, deadline_ts=None, allow_model=True) -> list[dict]:
    check_request_cancelled()
    signature = hashlib.sha256(repr((
        store.context_tokens, store.recent_tokens, store.summary_tokens,
        store.summary_input_tokens, store.history_limit,
        getattr(store.summarizer, 'signature', 'custom' if store.summarizer else 'rules'),
        'structured-v1',
    )).encode()).hexdigest()
    with store._transaction() as db:
        rows = db.execute(
            "SELECT id, created_at, payload FROM chat_memory_turns WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
        if not rows:
            return []
        db.execute("UPDATE chat_memory SET accessed_at=? WHERE session_id=?", (store.clock(), session_id))
        cached = db.execute(
            "SELECT cutoff, archive_floor, signature, content, method "
            "FROM chat_memory_compactions WHERE session_id=?", (session_id,),
        ).fetchone()
    turns = [{"id": n, "created_at": at, **json.loads(zlib.decompress(payload))}
             for n, at, payload in rows]
    floor, head = turns[0]['id'], turns[-1]['id']
    if cached and (cached[1] != floor or cached[2] != signature):
        cached = None
    cutoff = cached[0] if cached else 0
    recent = [turn for turn in turns if turn['id'] > cutoff]
    prefix = ([{"role": "memory", "content": cached[3], "compression": cached[4]}]
              if cached else [])
    # Reserve space for same-session recall and fixed API history instructions.
    available = store.context_tokens - store.recall_tokens - 768
    exact = _messages(recent, 2**63 - 1)
    if _size(prefix + exact) <= available:
        return prefix + exact
    if cached and len(recent) <= store.history_limit // 2:
        return prefix + _messages(recent, store.recent_tokens)
    tail = turns[-store.history_limit // 2:]
    older = turns[:-len(tail)]
    rendered_tail = _messages(tail, store.recent_tokens)
    if not older:
        note = [{"role": "memory", "content": "长消息已移出上下文；原文可在当前会话内按问题回查。",
                 "compression": "budget_offload"}]
        return note + rendered_tail
    source_turns = _sources(older, store.summary_input_tokens)
    content, method = _fallback(older, store.summary_tokens), 'rule_fallback'
    if allow_model and store.summarizer and source_turns:
        try:
            output = store.summarizer(json.dumps(source_turns, ensure_ascii=False), deadline_ts=deadline_ts)
            supplied_ids = {turn['id'] for turn in source_turns}
            content = validate_summary(output, [t for t in older if t['id'] in supplied_ids],
                                       store.summary_tokens)
            method = 'model_summary'
        except Exception:  # noqa: BLE001 - Optional model failure must retain a local fallback.
            check_request_cancelled()
    check_request_cancelled()
    # Never hold SQLite's write lock during inference. Reject stale snapshots after
    # concurrent append, expiry, pruning or clear/recreate; a summary cannot resurrect memory.
    stale = False
    with store._transaction() as db:
        current = db.execute(
            "SELECT MIN(id), MAX(id) FROM chat_memory_turns WHERE session_id=?", (session_id,),
        ).fetchone()
        if current == (floor, head):
            db.execute(
                "INSERT INTO chat_memory_compactions VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET cutoff=excluded.cutoff, "
                "archive_floor=excluded.archive_floor, signature=excluded.signature, "
                "content=excluded.content, method=excluded.method",
                (session_id, older[-1]['id'], floor, signature, content, method),
            )
        else:
            stale = True
    if stale and allow_model:
        # One re-read with deterministic fallback, no recursive model request.
        return build_context(store, session_id, deadline_ts=deadline_ts, allow_model=False)
    return [{"role": "memory", "content": content, "compression": method}] + rendered_tail
