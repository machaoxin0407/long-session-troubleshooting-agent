"""Bounded semantic embeddings for session trees, with no retries or downloads."""
from __future__ import annotations

import asyncio
import hashlib
import math
import os
import threading
import time
from contextlib import contextmanager, nullcontext

import httpx
import numpy as np

from request_cancellation import CURRENT_CANCELLATION, check_request_cancelled, wait_for_request_cancellation
from session_context import fit_text
from model_attempts import record_attempt

_SLOTS = threading.BoundedSemaphore(2)


class SessionEmbedder:
    def __init__(self, *, base_url=None, api_key=None, model=None, timeout_s=4.,
                 batch_size=32, text_bytes=8192, on_request=None):
        if (not math.isfinite(timeout_s) or timeout_s <= 0 or not 1 <= batch_size <= 128
                or text_bytes < 128):
            raise ValueError('invalid session embedding limits')
        def setting(name):
            return os.getenv('CHAT_SESSION_TREE_' + name) or os.getenv(name, '')
        self.base_url = (base_url if base_url is not None else setting('EMBEDDING_BASE_URL')).rstrip('/')
        self.api_key = api_key if api_key is not None else setting('EMBEDDING_API_KEY')
        self.model = model if model is not None else setting('EMBEDDING_MODEL')
        # Model Studio limits Qwen3.7 text embedding requests to 20 inputs.
        if self.model in {'qwen3.7-text-embedding', 'qwen3.7-text-embedding-flash'}:
            batch_size = min(batch_size, 20)
        self.timeout_s, self.batch_size, self.text_bytes = timeout_s, batch_size, text_bytes
        self.on_request = on_request
        self._local = threading.local()

    @contextmanager
    def connection_scope(self):
        """Reuse TLS only during one context build/query, never across sessions."""
        if getattr(self._local, 'loop', None) is not None:
            yield
            return
        loop = asyncio.new_event_loop()
        client = httpx.AsyncClient(timeout=None, trust_env=False)
        self._local.loop, self._local.client = loop, client
        try:
            yield
        finally:
            try:
                loop.run_until_complete(client.aclose())
            finally:
                self._local.loop = self._local.client = None
                loop.close()

    @property
    def signature(self):
        # Cache compatibility includes the endpoint without recording its private address.
        return hashlib.sha256(repr((self.base_url, self.model, self.text_bytes, 'session-embedding-v1')).encode()).hexdigest()

    @property
    def configured(self):
        return bool(self.base_url and self.api_key and self.model)

    def __call__(self, texts, *, deadline_ts=None):
        check_request_cancelled()
        if not self.configured:
            raise RuntimeError('session embedding configuration missing')
        if not texts:
            return []
        deadline = min(time.time() + self.timeout_s, deadline_ts or float('inf'))
        if deadline <= time.time() or not _SLOTS.acquire(blocking=False):
            raise TimeoutError('session embedding unavailable within budget')
        try:
            loop = getattr(self._local, 'loop', None)
            return (loop.run_until_complete(self._request(texts, deadline)) if loop is not None
                    else asyncio.run(self._request(texts, deadline)))
        finally:
            _SLOTS.release()

    async def _request(self, texts, deadline):
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError('session embedding deadline')

        async def exchange():
            vectors, dimension = [], None
            clipped = [fit_text(text, self.text_bytes) for text in texts]
            unique = list(dict.fromkeys(clipped))
            existing = getattr(self._local, 'client', None)
            async with (nullcontext(existing) if existing is not None else
                        httpx.AsyncClient(timeout=remaining, trust_env=False)) as client:
                for offset in range(0, len(unique), self.batch_size):
                    check_request_cancelled()
                    batch = unique[offset:offset + self.batch_size]
                    if self.on_request is not None:
                        self.on_request()
                    record_attempt('embedding')
                    response = await client.post(self.base_url + '/embeddings',
                                                 headers={'Authorization': 'Bearer ' + self.api_key},
                                                 json={'model': self.model, 'input': batch})
                    response.raise_for_status()
                    data = response.json()['data']
                    if (not isinstance(data, list) or len(data) != len(batch)
                            or any(not isinstance(item, dict) or type(item.get('index')) is not int for item in data)
                            or sorted(item['index'] for item in data) != list(range(len(batch)))):
                        raise ValueError('invalid embedding response indices')
                    array = np.asarray([item['embedding'] for item in sorted(data, key=lambda item: item['index'])],
                                       dtype=float)
                    if (array.ndim != 2 or array.shape[1] < 1 or not np.isfinite(array).all()
                            or (dimension is not None and dimension != array.shape[1])):
                        raise ValueError('invalid embedding vectors')
                    dimension = array.shape[1]
                    vectors.extend(array.tolist())
            by_text = dict(zip(unique, vectors))
            return [by_text[text] for text in clipped]

        task = asyncio.create_task(exchange())
        cancellation = CURRENT_CANCELLATION.get()
        watcher = (asyncio.create_task(wait_for_request_cancellation(cancellation))
                   if cancellation is not None else None)
        try:
            if watcher is None:
                return await asyncio.wait_for(task, remaining)
            done, _ = await asyncio.wait((task, watcher), timeout=remaining,
                                         return_when=asyncio.FIRST_COMPLETED)
            check_request_cancelled()
            if task not in done:
                raise TimeoutError('session embedding deadline')
            return task.result()
        finally:
            task.cancel()
            tasks = [task]
            if watcher is not None:
                watcher.cancel()
                tasks.append(watcher)
            await asyncio.gather(*tasks, return_exceptions=True)
