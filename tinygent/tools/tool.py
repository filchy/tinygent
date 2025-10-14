from typing import Any
from typing import Callable
from typing import Generic
from typing import Iterable
from typing import Literal
from typing import TypeVar
from typing import cast
from typing import overload

from tinygent.datamodels.tool import AbstractTool
from tinygent.datamodels.tool_info import ToolInfo
from tinygent.runtime.executors import run_async_in_executor
from tinygent.runtime.global_registry import GlobalRegistry
from tinygent.types.base import TinyModel
from tinygent.utils.schema_validator import validate_schema

T = TypeVar('T', bound=TinyModel)
R = TypeVar('R')


class ToolConfig(TinyModel):
    type: Literal['tool'] = 'tool'

    name: str


class Tool(AbstractTool, Generic[T, R]):
    def __init__(
        self,
        fn: Callable[[T], R],
        use_cache: bool = False,
        cache_size: int | None = None,
    ) -> None:
        self.__original_fn = fn

        self._cached_fn: Callable[[T], R] | None = None
        self._info: ToolInfo[T, R] = ToolInfo.from_callable(
            fn, use_cache=use_cache, cache_size=cache_size
        )

        if use_cache:
            if not self.info.is_cachable:
                raise ValueError(
                    'Caching is not supported for generator or async generator tools.'
                )

            cache_size = cache_size or 128
            if self.info.is_coroutine:
                from async_lru import alru_cache

                self._cached_fn = alru_cache(maxsize=cache_size)(fn)  # type: ignore[misc]
            else:
                from functools import lru_cache

                self._cached_fn = lru_cache(maxsize=cache_size)(fn)

        self._fn = self._cached_fn or self.__original_fn

    @property
    def info(self) -> ToolInfo[T, R]:
        return self._info

    def clear_cache(self) -> None:
        if self._cached_fn:
            clear = getattr(self._cached_fn, 'cache_clear', None)
            if callable(clear):
                clear()

    def cache_info(self) -> Any:
        if self._cached_fn:
            info = getattr(self._cached_fn, 'cache_info', None)
            if callable(info):
                return info()
        return None

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

            return run_async_in_executor(run_async_gen)

        elif self.info.is_coroutine:

            async def run_coroutine():
                return await self._fn(*parsed_args, **kwargs)  # type: ignore[misc]

            return run_async_in_executor(run_coroutine)

        else:
            result = self._fn(*parsed_args, **kwargs)  # type: ignore[misc]
            if self.info.is_generator:
                return list(cast(Iterable[Any], result))
            else:
                return result


@overload
def tool(fn: Callable[[T], R]) -> Tool[T, R]: ...


@overload
def tool(
    *, use_cache: bool = False, cache_size: int = 128, hidden: bool = False
) -> Callable[[Callable[[T], R]], Tool[T, R]]: ...


def tool(
    fn: Callable[[T], R] | None = None,
    *,
    use_cache: bool = False,
    cache_size: int = 128,
    hidden: bool = False,
) -> Tool[T, R] | Callable[[Callable[[T], R]], Tool[T, R]]:
    def wrapper(f: Callable[[T], R]) -> Tool[T, R]:
        tool_instance = Tool(f, use_cache=use_cache, cache_size=cache_size)
        GlobalRegistry.get_registry().register_tool(tool_instance, hidden=hidden)
        return tool_instance

    if fn is None:
        return wrapper
    return wrapper(fn)
