import asyncio
from collections.abc import Coroutine
from concurrent.futures import Future
import gc
import os
import threading
import typing
from typing import Any
from typing import Callable

P = typing.ParamSpec('P')
T = typing.TypeVar('T')

_bg_loop = None
_bg_thread = None

_DEFAULT_SEMAPHORE_LIMIT = int(os.getenv('TINY_SEMPATHORE_DEFAULT_LIMIT', 5))


def _ensure_background_loop():
    global _bg_loop, _bg_thread
    if _bg_loop is None:
        _bg_loop = asyncio.new_event_loop()
        _bg_thread = threading.Thread(target=_bg_loop.run_forever, daemon=True)
        _bg_thread.start()
    return _bg_loop


async def run_in_semaphore(
    *coroutines: Coroutine,
    max_coroutines: int | None = None,
):
    semaphore = asyncio.Semaphore(max_coroutines or _DEFAULT_SEMAPHORE_LIMIT)

    async def _wrap_coroutine(coroutine: Coroutine) -> Any:
        async with semaphore:
            return await coroutine

    return await asyncio.gather(
        *(_wrap_coroutine(coroutine) for coroutine in coroutines)
    )


async def run_sync_in_executor(
    func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    def _inner() -> T:
        try:
            return func(*args, **kwargs)
        except StopIteration as exc:
            # StopIteration can't be set on an asyncio.Future
            # it raises a TypeError and leaves the Future pending forever
            # so we need to convert it to a RuntimeError
            raise RuntimeError from exc

    return await asyncio.get_running_loop().run_in_executor(None, _inner)


def run_async_in_executor(
    func: Callable[P, Coroutine[Any, Any, T]], *args: P.args, **kwargs: P.kwargs
) -> T:
    """Run an async function in a blocking manner.

    If called from a sync context (no running loop), creates and runs a new loop.
    If called from an async context (existing loop), schedules on a background thread.
    """
    coro = func(*args, **kwargs)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # no running loop -> create one and wait for all tasks to complete
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(coro)
            # Keep gathering pending tasks until none remain
            # This ensures cleanup tasks (like httpx client cleanup) are completed
            # We need multiple iterations because cleanup can spawn new tasks
            max_iterations = 10
            for _ in range(max_iterations):
                pending = asyncio.all_tasks(loop)
                if not pending:
                    break
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            return result
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                # Gather any tasks created during shutdown_asyncgens
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.run_until_complete(loop.shutdown_default_executor())
                # Final cleanup: gather any remaining tasks before closing
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                # Force garbage collection to trigger any __del__ methods
                # This ensures httpx AsyncClient and similar objects clean up before loop closes
                gc.collect()
                # Wait for any cleanup tasks created by garbage collection
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            finally:
                asyncio.set_event_loop(None)
                loop.close()
    else:
        # already inside a loop -> schedule onto background loop
        loop = _ensure_background_loop()
        future: Future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
