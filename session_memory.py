"""Bounded, persistent memory for one chat session; no cross-session lookup.

Legacy stores use local excerpts. API stores can enable token-budgeted context
with a source-validated model summary and local fallback. Recall never crosses sessions.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import zlib
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

_OMITTED = "\n[中间内容已省略]\n"
_IMPORTANT = re.compile(
    r"型号|错误|故障|不能|不要|没有|更正|不是|尚未|未完成|已经|已尝试|不需要|"
    r"\b(?:not|never|without|failed|error|model|instead|correction)\b|"
    r"\b[A-Z][A-Z0-9]*[-_]?\d+[A-Z0-9-]*\b", re.IGNORECASE,
)


def observe(text: str, limit: int) -> str:
    """Prefer intact sentences with identifiers, constraints or corrections.

    This is a source excerpt, not a semantic fact extractor. Selected sentences
    remain in their original order; gaps are explicit to avoid joining clauses.
    """
    if len(text) <= limit:
        return text
    sentences = [s for s in re.split(r"(?<=[。！？])|(?<=[.!?])\s+|\n+", text) if s]
    ranked = sorted(range(len(sentences)), key=lambda i: (
        bool(_IMPORTANT.search(sentences[i])), i == 0 or i == len(sentences) - 1
    ), reverse=True)
    selected = []
    for index in ranked:
        candidate = sorted(selected + [index])
        rendered = _OMITTED.join(sentences[i] for i in candidate)
        if len(rendered) + 2 * len(_OMITTED) <= limit:
            selected = candidate
    if not selected:
        return excerpt(text, limit)
    return excerpt(_OMITTED + _OMITTED.join(sentences[i] for i in selected) + _OMITTED, limit)


def _terms(text: str) -> set[str]:
    """Small lexical index for same-session archive recall, including CJK pairs."""
    result = set(re.findall(r"[a-z0-9][a-z0-9_-]{1,}", text.lower()))
    for segment in re.findall(r"[\u4e00-\u9fff]+", text):
        result.update(segment[i:i + 2] for i in range(len(segment) - 1))
    return result


def excerpt(text: str, limit: int) -> str:
    """Keep whole leading/trailing sentences when possible; mark every omission."""
    if len(text) <= limit:
        return text
    available = limit - len(_OMITTED)
    head_size = available * 2 // 3
    tail_size = available - head_size
    boundaries = [m.end() for m in re.finditer(r"[。！？]|[.!?](?:\s|$)|\n", text)]
    head_end = max((n for n in boundaries if n <= head_size), default=head_size)
    tail_start = next((n for n in boundaries if n >= len(text) - tail_size), len(text) - tail_size)
    return text[:head_end] + _OMITTED + text[tail_start:]


class SessionMemoryStore:
    """SQLite transactions merge concurrent appends, including across processes.

    Only successful turns are appended by the API. TTL is measured from that write;
    reads update LRU order but do not prolong retention. Construction performs no IO.
    """

    def __init__(
        self, path: str | Path, *, history_limit: int = 6,
        max_sessions: int = 500, ttl_s: float = 3600,
        recent_chars: int = 6000, summary_chars: int = 4000,
        excerpt_chars: int = 800, recall_chars: int = 2000,
        archive_turns: int = 200, archive_chars: int = 2_000_000,
        clock: Callable[[], float] = time.time,
        context_tokens: int | None = None, recent_tokens: int = 2800,
        summary_tokens: int = 1400, recall_tokens: int = 800,
        summary_input_tokens: int = 16000, summarizer: Callable | None = None,
        tree_enabled: bool = False, tree_retrieval: str = 'collapsed',
        tree_settings=None, tree_summarizer: Callable | None = None,
        tree_embedder: Callable | None = None,
    ):
        if history_limit < 2 or history_limit % 2:
            raise ValueError("history_limit must be a positive even message count")
        if max_sessions < 1 or not 0 < ttl_s < float("inf"):
            raise ValueError("max_sessions and ttl_s must be positive and finite")
        if min(summary_chars, excerpt_chars, recent_chars // history_limit) < 64:
            raise ValueError("each memory excerpt budget must be at least 64 characters")
        if recall_chars < 384 or archive_turns < history_limit // 2 or archive_chars < 128:
            raise ValueError("invalid archive or recall budget")
        self.path = Path(path)
        self.history_limit = history_limit
        self.max_sessions = max_sessions
        self.ttl_s = ttl_s
        self.recent_chars = recent_chars
        self.summary_chars = summary_chars
        self.excerpt_chars = min(excerpt_chars, summary_chars)
        self.recall_chars = recall_chars
        self.archive_turns = archive_turns
        self.archive_chars = archive_chars
        self.clock = clock
        if context_tokens is not None and (
            min(recent_tokens, summary_tokens, recall_tokens) < 128
            or recent_tokens < history_limit * 160
            or recent_tokens + summary_tokens + recall_tokens + 768 > context_tokens
            or summary_input_tokens < summary_tokens
        ):
            raise ValueError("invalid token memory budgets")
        self.context_tokens = context_tokens
        self.recent_tokens = recent_tokens
        self.summary_tokens = summary_tokens
        self.recall_tokens = recall_tokens
        self.summary_input_tokens = summary_input_tokens
        self.summarizer = summarizer
        from session_tree import TreeSettings

        if tree_retrieval not in {'collapsed', 'traversal'}:
            raise ValueError('invalid tree retrieval mode')
        self.tree_settings = tree_settings or TreeSettings()
        if tree_enabled and (context_tokens is None or self.tree_settings.max_leaves < archive_turns * 2):
            raise ValueError('tree memory requires token budgets and max_leaves >= archive_turns * 2')
        self.tree_enabled = tree_enabled
        self.tree_retrieval = tree_retrieval
        self.tree_summarizer = tree_summarizer
        self.tree_embedder = tree_embedder

    @contextmanager
    def _transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS chat_memory ("
                "session_id TEXT PRIMARY KEY, recent TEXT NOT NULL, summary TEXT NOT NULL, "
                "updated_at REAL NOT NULL, accessed_at REAL NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS chat_memory_expiry ON chat_memory(updated_at)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS chat_memory_turns ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
                "created_at REAL NOT NULL, payload BLOB NOT NULL, chars INTEGER NOT NULL, "
                "FOREIGN KEY(session_id) REFERENCES chat_memory(session_id) ON DELETE CASCADE)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS chat_memory_turn_session ON chat_memory_turns(session_id, id)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS chat_memory_compactions ("
                "session_id TEXT PRIMARY KEY REFERENCES chat_memory(session_id) ON DELETE CASCADE, "
                "cutoff INTEGER NOT NULL, archive_floor INTEGER NOT NULL, signature TEXT NOT NULL, "
                "content TEXT NOT NULL, method TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS chat_memory_trees ("
                "session_id TEXT PRIMARY KEY REFERENCES chat_memory(session_id) ON DELETE CASCADE, "
                "signature TEXT NOT NULL, payload BLOB NOT NULL)"
            )
            connection.execute(
                "DELETE FROM chat_memory WHERE updated_at <= ?", (self.clock() - self.ttl_s,)
            )
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(self, session_id: str, *, deadline_ts: float | None = None, question: str = '',
            retrieval_mode: str | None = None) -> list[dict]:
        if self.tree_enabled:
            from session_tree import build_tree_context

            return build_tree_context(self, session_id, question, retrieval_mode=retrieval_mode,
                                      deadline_ts=deadline_ts)
        if self.context_tokens is not None:
            from session_context import build_context

            return build_context(self, session_id, deadline_ts=deadline_ts)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT recent, summary FROM chat_memory WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return []
            connection.execute(
                "UPDATE chat_memory SET accessed_at = ? WHERE session_id = ?",
                (self.clock(), session_id),
            )
            recent, summary = json.loads(row[0]), json.loads(row[1])
            # Reapply budgets if an operator reduces them after restarting.
            summary, recent = self._compact(summary, recent, connection)
            if summary:
                labels = {"user": "用户原话摘录", "assistant": "历史客服回答摘录（未重新核验）"}
                compressed = "\n".join(
                    f"{labels[item['role']]}{item.get('source', '')}: {item['content']}"
                    for item in summary
                )
                return [{"role": "memory", "content": compressed}] + recent
            older = connection.execute(
                "SELECT COUNT(*) FROM chat_memory_turns WHERE session_id=?", (session_id,)
            ).fetchone()[0] > len(recent) // 2
            if older:
                return [{"role": "memory", "content": "较早对话已存档；摘要预算不足，需按当前问题回查。"}] + recent
            if any(_OMITTED in item['content'] for item in recent):
                return [{"role": "memory", "content": "近期长消息已按预算摘录，省略细节可从当前会话原文回查。"}] + recent
            return recent

    def _compact(self, summary: list[dict], recent: list[dict], connection=None) -> tuple[list[dict], list[dict]]:
        if len(recent) > self.history_limit:
            evicted = recent[:-self.history_limit]
            # Build observations from the canonical original, never from an
            # already clipped recent message (which could omit a middle constraint).
            for item in evicted:
                if connection is not None and item.get("turn_id"):
                    row = connection.execute(
                        "SELECT payload FROM chat_memory_turns WHERE id=?", (item["turn_id"],)
                    ).fetchone()
                    if row:
                        raw = json.loads(zlib.decompress(row[0]))
                        item = {**item, "content": raw["question" if item["role"] == "user" else "answer"]}
                summary = summary + [item]
            recent = recent[-self.history_limit:]
        summary = [
            {**item, "content": observe(item["content"], self.excerpt_chars)}
            for item in summary
        ]
        # Include role labels/newlines in the summary budget, not only content.
        def size(items):
            return sum(len(item["content"]) + len(item.get("source", "")) + 32 for item in items)

        while summary and size(summary) > self.summary_chars:
            # Prefer user constraints/corrections to old unverified assistant prose.
            victim = min(range(len(summary)), key=lambda i: (
                summary[i]["role"] == "user", bool(_IMPORTANT.search(summary[i]["content"])), i
            ))
            summary.pop(victim)
        recent = [
            {**item, "content": observe(
                item["content"], self.recent_chars // self.history_limit
            )}
            for item in recent
        ]
        return summary, recent

    def append(self, session_id: str, question: str, answer: str) -> None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT recent, summary FROM chat_memory WHERE session_id = ?", (session_id,)
            ).fetchone()
            recent, summary = (json.loads(row[0]), json.loads(row[1])) if row else ([], [])
            now = self.clock()
            if row is None:
                connection.execute("INSERT INTO chat_memory VALUES (?, '[]', '[]', ?, ?)", (session_id, now, now))
            # Canonical raw transcript is retained separately from prompt compression.
            # Extreme oversized turns are explicitly clipped to the archive limit.
            raw_question, raw_answer = question, answer
            if len(question) + len(answer) > self.archive_chars:
                raw_question = excerpt(question, self.archive_chars // 2)
                raw_answer = excerpt(answer, self.archive_chars // 2)
            payload = json.dumps({"question": raw_question, "answer": raw_answer}, ensure_ascii=False)
            cursor = connection.execute(
                "INSERT INTO chat_memory_turns(session_id, created_at, payload, chars) VALUES (?, ?, ?, ?)",
                (session_id, now, zlib.compress(payload.encode("utf-8")), len(raw_question) + len(raw_answer)),
            )
            source = f" [turn={cursor.lastrowid}; unix={now:.3f}]"
            recent.extend([
                {"role": "user", "content": question, "source": source, "turn_id": cursor.lastrowid},
                {"role": "assistant", "content": answer, "source": source, "turn_id": cursor.lastrowid},
            ])
            summary, recent = self._compact(summary, recent, connection)
            connection.execute(
                "INSERT INTO chat_memory VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET recent=excluded.recent, "
                "summary=excluded.summary, updated_at=excluded.updated_at, accessed_at=excluded.accessed_at",
                (session_id, json.dumps(recent, ensure_ascii=False),
                 json.dumps(summary, ensure_ascii=False), now, now),
            )
            connection.execute(
                "DELETE FROM chat_memory WHERE session_id IN "
                "(SELECT session_id FROM chat_memory ORDER BY accessed_at DESC, rowid DESC LIMIT -1 OFFSET ?)",
                (self.max_sessions,),
            )
            archived = connection.execute(
                "SELECT id, chars FROM chat_memory_turns WHERE session_id=? ORDER BY id DESC", (session_id,)
            ).fetchall()
            total = 0
            for index, (turn_id, chars) in enumerate(archived):
                total += chars
                if index >= self.archive_turns or total > self.archive_chars:
                    connection.execute(
                        "DELETE FROM chat_memory_turns WHERE session_id=? AND id<=?", (session_id, turn_id)
                    )
                    break

    def recall(self, session_id: str, question: str, *, visible_history=None) -> list[dict[str, str]]:
        """Recall up to three relevant old turns from this session's raw archive."""
        query = _terms(question)
        if not query:
            return []
        with self._transaction() as connection:
            state = connection.execute(
                "SELECT recent FROM chat_memory WHERE session_id=?", (session_id,)
            ).fetchone()
            recent = (visible_history if visible_history is not None else
                      json.loads(state[0])[-self.history_limit:] if state else [])
            visible = {item.get('turn_id') for item in recent}
            clipped = {item.get('turn_id') for item in recent
                       if _OMITTED in item['content'] or item.get('offloaded')}
            rows = connection.execute(
                "SELECT id, created_at, payload FROM chat_memory_turns WHERE session_id=? "
                "ORDER BY id DESC", (session_id,),
            ).fetchall()
            candidates = []
            for turn_id, created_at, payload in rows:
                if turn_id in visible and turn_id not in clipped:
                    continue
                turn = json.loads(zlib.decompress(payload))
                overlap = query & _terms(turn["question"] + " " + turn["answer"])
                if overlap:
                    candidates.append((len(overlap), turn_id, created_at, turn))
            selected = sorted(candidates, reverse=True)[:3]
            if not selected:
                return []
            if self.context_tokens is not None:
                from session_context import render_recall

                content = render_recall(selected, question, self.recall_tokens - 80)
                return [{"role": "recall", "content": content}] if content else []
            budget = self.recall_chars // (2 * len(selected))
            lines = []
            for _, turn_id, created_at, turn in sorted(selected, key=lambda item: item[1]):
                lines.append(
                    f"[turn={turn_id}; unix={created_at:.3f}] 用户原话: {observe(turn['question'], budget)}\n"
                    f"历史客服回答（未重新核验）: {observe(turn['answer'], budget)}"
                )
            content = excerpt("\n".join(lines), self.recall_chars)
            return [{"role": "recall", "content": content}]

    def clear(self, session_id: str) -> None:
        with self._transaction() as connection:
            connection.execute("DELETE FROM chat_memory WHERE session_id = ?", (session_id,))
