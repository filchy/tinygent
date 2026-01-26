from io import StringIO
import logging
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Coroutine
from typing import Generic
from typing import Iterable
from typing import Literal
from typing import TypeVar
from typing import cast
from typing import overload

from pydantic import Field

from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.datamodels.tool import AbstractToolConfig
from tinygent.core.datamodels.tool_info import ToolInfo
from tinygent.core.runtime.executors import run_async_in_executor
from tinygent.core.runtime.tool_catalog import GlobalToolCatalog
from tinygent.utils.schema_validator import validate_schema

logger = logging.getLogger(__name__)

R = TypeVar('R')


class ToolConfig(AbstractToolConfig['Tool[R]'], Generic[R]):
    """Configuration for simple tools."""

    type: Literal['simple'] = Field(default='simple', frozen=True)

    def build(self) -> 'Tool[R]':
        raw_tool = GlobalToolCatalog().get_active_catalog().get_tool(self.name)

        return Tool(
            raw_tool.raw,
            use_cache=raw_tool.info.use_cache,
            cache_size=raw_tool.info.cache_size,
        )


class Tool(AbstractTool, Generic[R]):
    """A simple tool wrapping a callable function for agent use.

    Wraps Python functions (sync, async, generators, async generators) into
    a standardized tool interface that agents can invoke. Automatically extracts
    metadata like name, description, and parameter schemas from the function.

    The tool handles:
    - Input validation using Pydantic schemas
    - Automatic async/sync adaptation
    - Optional LRU caching for expensive operations
    - Generator/async generator collection into lists

    Caching is supported for regular and async functions (not generators).
    When enabled, results are memoized using LRU cache with configurable size.

    Args:
        fn: The callable function to wrap (sync/async, regular/generator)
        use_cache: Enable LRU caching of results (default: False)
        cache_size: Maximum cache size if caching enabled (default: None)
    """

    def __init__(
        self,
        fn: Callable[..., R],
        use_cache: bool = False,
        cache_size: int | None = None,
    ) -> None:
        self.__original_fn = fn

        self._cached_fn: Callable[..., Any] | Callable[..., Awaitable[Any]] | None = None
        self._info: ToolInfo[R] = ToolInfo.from_callable(
            fn, use_cache=use_cache, cache_size=cache_size
        )

        if self.info.is_generator or self.info.is_async_generator:
            use_cache = False
            self.info.use_cache = False

        if use_cache:
            if not self.info.is_cachable:
                raise ValueError(
                    'Caching is not supported for generator or async generator tools.'
                )

            cache_size = cache_size or 128
            if self.info.is_coroutine:
                from async_lru import alru_cache

                async_fn = cast(Callable[..., Coroutine[Any, Any, Any]], fn)
                self._cached_fn = alru_cache(maxsize=cache_size)(async_fn)
            else:
                from functools import lru_cache

                self._cached_fn = lru_cache(maxsize=cache_size)(fn)

        self._fn = self._cached_fn or self.__original_fn

    @property
    def raw(self) -> Callable[..., R]:
        return self.__original_fn

    @property
    def info(self) -> ToolInfo[R]:
        return self._info

    def clear_cache(self) -> None:
        if self._cached_fn:
            clear = getattr(self._cached_fn, 'cache_clear', None)
            if callable(clear):
                clear()

            logger.debug('Clearing cache for tool: %s', self.info.name)

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

        # If using auto-generated schema from regular params, unpack model to kwargs
        if (
            self.info.uses_auto_schema
            and parsed_args
            and hasattr(parsed_args[0], 'model_dump')
        ):
            kwargs = parsed_args[0].model_dump()
            parsed_args = []

        logger.debug(
            'Running tool: %s with args: %s, kwargs: %s',
            self.info.name,
            parsed_args,
            kwargs,
        )

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

    def __str__(self) -> str:
        buf = StringIO()

        buf.write(f'Tool - {self.info.name}\n')
        buf.write(f'\tDescription: {self.info.description}\n')
        buf.write(f'\tNumber of Args: {self.info.arg_count}\n')

        return buf.getvalue()


@overload
def tool(fn: Callable[..., R]) -> Tool[R]: ...


@overload
def tool(
    *, use_cache: bool = False, cache_size: int = 128
) -> Callable[[Callable[..., R]], Tool[R]]: ...


def tool(
    fn: Callable[..., R] | None = None,
    *,
    use_cache: bool = False,
    cache_size: int = 128,
) -> Tool[R] | Callable[[Callable[..., R]], Tool[R]]:
    def wrapper(f: Callable[..., R]) -> Tool[R]:
        tool_instance = Tool(f, use_cache=use_cache, cache_size=cache_size)
        return tool_instance

    if fn is None:
        return wrapper
    return wrapper(fn)


@overload
def register_tool(fn: Callable[..., Any]) -> Tool[Any]: ...


@overload
def register_tool(
    *, use_cache: bool = False, cache_size: int = 128, hidden: bool = False
) -> Callable[[Callable[..., Any]], Tool[Any]]: ...


def register_tool(
    fn: Callable[..., Any] | None = None,
    *,
    use_cache: bool = False,
    cache_size: int = 128,
    hidden: bool = False,
) -> Tool[Any] | Callable[[Callable[..., Any]], Tool[Any]]:
    def wrapper(f: Callable[..., Any]) -> Tool[Any]:
        GlobalToolCatalog().get_active_catalog().register(
            f, use_cache=use_cache, cache_size=cache_size, hidden=hidden
        )
        return Tool(
            f,
            use_cache=use_cache,
            cache_size=cache_size,
        )

    if fn is None:
        return wrapper
    return wrapper(fn)
