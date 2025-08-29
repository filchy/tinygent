import asyncio
import typing

from typing import Callable


P = typing.ParamSpec('P')
T = typing.TypeVar('T')


async def run_in_executor(
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs
) -> T:

    async def _inner() -> T:
        try:
            return func(*args, **kwargs)
        except StopIteration as exc:
            # StopIteration can't be set on an asyncio.Future
            # it raises a TypeError and leaves the Future pending forever
            # so we need to convert it to a RuntimeError
            raise RuntimeError from exc

    return await asyncio.get_running_loop().run_in_executor(None, _inner)
