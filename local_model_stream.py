"""Request-owned loopback SSE transport, cancellable even between tokens."""
import asyncio
import json
import time
from urllib.parse import urlparse

import httpx
from openai.types.chat import ChatCompletionChunk

from request_cancellation import (
    CURRENT_CANCELLATION,
    check_request_cancelled,
    wait_for_request_cancellation,
)


def uses_request_local_transport(base_url):
    url = urlparse(base_url)
    return (CURRENT_CANCELLATION.get() is not None and url.scheme in ('http', 'https')
            and url.hostname in ('127.0.0.1', 'localhost', '::1'))


async def _stream(*, base_url, api_key, deadline_ts, arguments, on_chunk):
    check_request_cancelled()
    remaining = deadline_ts - time.time()
    if remaining <= 0:
        raise TimeoutError('Local model deadline expired')
    body = dict(arguments)
    extra = body.pop('extra_body', {}) or {}
    if set(extra) & set(body):
        raise ValueError('Extra body shadows model request fields')
    headers = dict(body.pop('extra_headers', {}) or {})
    if any(key.lower() == 'authorization' for key in headers):
        raise ValueError('Request headers cannot override route authorization')
    body.update(extra)
    body['stream'] = True
    headers['Authorization'] = 'Bearer ' + api_key

    async def exchange():
        async with (httpx.AsyncClient(timeout=remaining, trust_env=False) as client,
                    client.stream('POST', base_url.rstrip('/') + '/chat/completions',
                                  json=body, headers=headers) as response):
            response.raise_for_status()
            data = []

            def dispatch():
                if not data:
                    return False
                payload = '\n'.join(data)
                data.clear()
                if payload.strip() == '[DONE]':
                    return True
                check_request_cancelled()
                on_chunk(ChatCompletionChunk(**json.loads(payload)))
                return False

            async for line in response.aiter_lines():
                check_request_cancelled()
                if not line:
                    if dispatch():
                        return
                elif line.startswith('data:'):
                    data.append(line[5:].removeprefix(' '))
            dispatch()

    task = asyncio.create_task(exchange())
    event = CURRENT_CANCELLATION.get()
    watcher = asyncio.create_task(wait_for_request_cancellation(event)) if event is not None else None
    tasks = [task] + ([watcher] if watcher is not None else [])
    try:
        done, _ = await asyncio.wait(tasks, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
        check_request_cancelled()
        if task not in done:
            raise TimeoutError('Local model deadline expired')
        return task.result()
    finally:
        for pending in tasks:
            pending.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def stream_local_model_until(*, base_url, api_key, deadline_ts, arguments, on_chunk):
    url = urlparse(base_url)
    if url.scheme not in ('http', 'https') or url.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Streaming transport requires a loopback endpoint')
    return asyncio.run(_stream(base_url=base_url, api_key=api_key, deadline_ts=deadline_ts,
                               arguments=arguments, on_chunk=on_chunk))
