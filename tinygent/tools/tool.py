import asyncio

from typing import Any
from typing import Callable

from tinygent.datamodels.tool import AbstractTool
from tinygent.datamodels.tool_info import ToolInfo


class Tool(AbstractTool):

    def __init__(self, fn: Callable[..., Any]) -> None:

        self._fn = fn
        self._info = ToolInfo.from_callable(fn)

    @property
    def info(self) -> ToolInfo:

        return self._info

    def __call__(self, *args: Any, **kwargs: Any) -> Any:

        return self._fn(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:

        if self.info.is_async_generator:
            async def run_async_gen():
                result = []
                async for item in self._fn(*args, **kwargs):
                    result.append(item)
                return result
            return asyncio.run(run_async_gen())

        elif self.info.is_coroutine:
            async def run_coroutine():
                return await self._fn(*args, **kwargs)
            return asyncio.run(run_coroutine())

        else:
            result = self._fn(*args, **kwargs)
            if self.info.is_generator:
                return list(result)
            else:
                return result


def tool(fn: Callable[..., Any]) -> Tool:

    return Tool(fn)
