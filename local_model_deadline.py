"""One loopback model request with cancellable total wall-time budget."""
import asyncio
import time
from urllib.parse import urlparse

import httpx
from openai.types.chat import ChatCompletion

from request_cancellation import CURRENT_CANCELLATION, wait_for_request_cancellation


async def _request(*, base_url, api_key, deadline_ts, arguments, trace_events=None):
    cancellation = CURRENT_CANCELLATION.get()
    if cancellation is not None and cancellation.is_set():
        raise TimeoutError('Local model request cancelled')
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
    if body.get('stream', False):
        raise ValueError('Total-deadline call expects a complete response')
    headers['Authorization'] = 'Bearer ' + api_key
    started = time.monotonic()

    def record(event):
        if trace_events is not None:
            trace_events.append({'event': event, 'elapsed_seconds': time.monotonic() - started})

    async def trace(event, _info):
        # Deliberately omit HTTP trace values: they can contain headers or bodies.
        if event in {
            phase + '.' + state
            for phase in ('connection.connect_tcp', 'http11.send_request_headers',
                          'http11.send_request_body', 'http11.receive_response_headers',
                          'http11.receive_response_body', 'http11.response_closed')
            for state in ('started', 'complete', 'failed')
        }:
            record(event)
    # A read timeout is an inactivity timeout. The outer cancellation bounds
    # the entire response, including a body that keeps trickling. The client is
    # isolated per call and does not close another request's connection.
    # Direct async HTTP also avoids SDK startup work in non-cancellable threads.
    async def exchange():
        async with httpx.AsyncClient(timeout=remaining, trust_env=False) as client:
            response = await client.post(base_url.rstrip('/') + '/chat/completions',
                                         json=body, headers=headers,
                                         extensions={'trace': trace} if trace_events is not None else None)
            response.raise_for_status()
            return ChatCompletion(**response.json())

    record('request.started')
    exchange_task = asyncio.create_task(exchange())
    cancel_task = (asyncio.create_task(wait_for_request_cancellation(cancellation))
                   if cancellation is not None else None)
    try:
        if cancel_task is None:
            result = await asyncio.wait_for(exchange_task, timeout=remaining)
        else:
            done, _ = await asyncio.wait((exchange_task, cancel_task), timeout=remaining,
                                         return_when=asyncio.FIRST_COMPLETED)
            if cancellation.is_set():
                raise TimeoutError('Local model request cancelled')
            if exchange_task not in done:
                raise TimeoutError('Local model deadline expired')
            result = exchange_task.result()
    except BaseException:
        record('request.failed')
        raise
    finally:
        exchange_task.cancel()
        pending = [exchange_task]
        if cancel_task is not None:
            cancel_task.cancel()
            pending.append(cancel_task)
        await asyncio.gather(*pending, return_exceptions=True)
    record('request.complete')
    return result


def call_local_model_until(*, base_url, api_key, deadline_ts, arguments, trace_events=None):
    url = urlparse(base_url)
    if url.scheme not in ('http', 'https') or url.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Total-deadline call requires a loopback endpoint')
    if deadline_ts is None or deadline_ts <= time.time():
        raise TimeoutError('Local model deadline expired')
    return asyncio.run(_request(base_url=base_url, api_key=api_key,
                               deadline_ts=deadline_ts, arguments=arguments, trace_events=trace_events))
