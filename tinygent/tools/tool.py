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
from tinygent.runtime.global_registry import GlobalRegistry
from tinygent.utils.schema_validator import validate_schema

T = TypeVar('T', bound=BaseModel)
R = TypeVar('R')


class Tool(AbstractTool, Generic[T, R]):
    def __init__(self, fn: Callable[[T], R]) -> None:
        self._fn: Callable[[T], R] = fn
        self._info: ToolInfo[T, R] = ToolInfo.from_callable(fn)

    @property
    def info(self) -> ToolInfo[T, R]:
        return self._info

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(*args, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        parsed_args: list[Any] = list(args)

        if self.info.input_schema is not None:
            input_model_cls = self.info.input_schema

            if parsed_args and isinstance(parsed_args[0], dict):
                parsed_args[0] = validate_schema(parsed_args[0], input_model_cls)

            elif not parsed_args and kwargs:
                parsed_args = [validate_schema(kwargs, input_model_cls)]
                kwargs = {}

        if self.info.is_async_generator:

            async def run_async_gen():
                result = []
                async for item in self._fn(*parsed_args, **kwargs):  # type: ignore[misc]
                    result.append(item)

                return result

            return asyncio.run(run_async_gen())

        elif self.info.is_coroutine:

            async def run_coroutine():
                return await self._fn(*parsed_args, **kwargs)  # type: ignore[misc]

            return asyncio.run(run_coroutine())

        else:
            result = self._fn(*parsed_args, **kwargs)  # type: ignore[misc]
            if self.info.is_generator:
                return list(cast(Iterable[Any], result))
            else:
                return result


def tool(fn: Callable[[T], R]) -> Tool[T, R]:
    tool_fn = Tool(fn)

    GlobalRegistry.get_registry().register_tool(tool_fn)

    return tool_fn
