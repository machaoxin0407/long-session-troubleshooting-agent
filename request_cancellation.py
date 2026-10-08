"""Request-owned cooperative cancellation across async and worker boundaries."""
import asyncio
import contextvars
import threading
import time

CURRENT_CANCELLATION = contextvars.ContextVar('rag3d_request_cancellation', default=None)


class ClientDisconnectedError(ConnectionError):
    """The caller disconnected while its worker was running."""


def check_request_cancelled():
    event = CURRENT_CANCELLATION.get()
    if event is not None and event.is_set():
        raise TimeoutError('Request cancelled')


def acquire_request_slot(semaphore, timeout):
    """Retain legacy wait policy while making request-owned waits interruptible."""
    event = CURRENT_CANCELLATION.get()
    if event is None:
        return semaphore.acquire() if timeout <= 0 else semaphore.acquire(timeout=timeout)
    end = time.monotonic() + timeout if timeout > 0 else None
    while True:
        check_request_cancelled()
        remaining = end - time.monotonic() if end is not None else 0.05
        if remaining <= 0:
            return False
        if semaphore.acquire(timeout=min(remaining, 0.05)):
            if event.is_set():
                semaphore.release()
                check_request_cancelled()
            return True


async def run_request_worker(request, executor, function, args, timeout):
    event = threading.Event()
    token = CURRENT_CANCELLATION.set(event)
    try:
        context = contextvars.copy_context()
    finally:
        CURRENT_CANCELLATION.reset(token)
    worker = asyncio.get_running_loop().run_in_executor(executor, context.run, function, *args)

    async def disconnected():
        # Some ASGI receive implementations use a cancellation scope for
        # nonblocking polling. A task.cancel() during that scope may be
        # consumed, so termination must also have its own explicit signal.
        while not event.is_set():
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.05)

    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((worker, watcher), timeout=timeout,
                                     return_when=asyncio.FIRST_COMPLETED)
        if worker in done:
            return worker.result()
        if watcher in done:
            watcher.result()
            raise ClientDisconnectedError('Client disconnected')
        raise TimeoutError('Request deadline expired')
    finally:
        event.set()
        worker.cancel()
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)


async def wait_for_request_cancellation(event):
    while not event.is_set():
        await asyncio.sleep(0.02)
