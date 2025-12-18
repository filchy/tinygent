import os
import asyncio
from collections.abc import Coroutine
from concurrent.futures import Future
import threading
import typing
from typing import Any
from typing import Callable

P = typing.ParamSpec('P')
T = typing.TypeVar('T')

_bg_loop = None
_bg_thread = None

SEMAPHORE_LIMIT = int(os.getenv('TINY_SEMPATHORE_LIMIT', 5))


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
    semaphore = asyncio.Semaphore(max_coroutines or SEMAPHORE_LIMIT)

    async def _wrap_coroutine(coroutine: Coroutine) -> Any:
        async with semaphore:
            return await coroutine

    return await asyncio.gather(*(_wrap_coroutine(coroutine) for coroutine in coroutines))


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
    coro = func(*args, **kwargs)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # no running loop → we can block here
        return asyncio.run(coro)
    else:
        # already inside a loop → schedule onto background loop
        loop = _ensure_background_loop()
        future: Future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
