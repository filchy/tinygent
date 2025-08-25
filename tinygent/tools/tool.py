import asyncio

from typing import Any
from typing import Callable
from typing import Generic
from typing import Iterable
from typing import TypeVar
from typing import cast
from pydantic import BaseModel

from tinygent.datamodels.tool import AbstractTool
from tinygent.datamodels.tool_info import ToolInfo

T = TypeVar('T', bound=BaseModel)
R = TypeVar('R')


class Tool(AbstractTool, Generic[T, R]):

    def __init__(self, fn: Callable[[T], R]) -> None:

        self._fn: Callable[[T], R] = fn
        self._info: ToolInfo[T, R] = ToolInfo.from_callable(fn)

    @property
    def info(self) -> ToolInfo[T, R]:

        return self._info

    def __call__(self, data: T) -> R:

        return self._fn(data)

    def run(self, *args: Any, **kwargs: Any) -> Any:

        if self.info.is_async_generator:
            async def run_async_gen():

                result = []
                async for item in self._fn(*args, **kwargs):  # type: ignore[misc]
                    result.append(item)

                return result

            return asyncio.run(run_async_gen())

        elif self.info.is_coroutine:
            async def run_coroutine():

                return await self._fn(*args, **kwargs)  # type: ignore[misc]

            return asyncio.run(run_coroutine())

        else:
            result = self._fn(*args, **kwargs)  # type: ignore[misc]
            if self.info.is_generator:
                return list(cast(Iterable[Any], result))
            else:
                return result


def tool(fn: Callable[[T], R]) -> Tool[T, R]:

    return Tool(fn)
