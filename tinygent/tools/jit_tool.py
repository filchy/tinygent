from collections.abc import Generator
import inspect
from io import StringIO
from types import GeneratorType
from typing import Any
from typing import Callable
from typing import Generic
from typing import Literal
from typing import TypeVar
from typing import overload

from pydantic import Field

from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.datamodels.tool import AbstractToolConfig
from tinygent.core.datamodels.tool_info import ToolInfo
from tinygent.core.runtime.tool_catalog import GlobalToolCatalog
from tinygent.core.types.base import TinyModel
from tinygent.tools.tool import Tool

T = TypeVar('T', bound=TinyModel)


class JITInstructionToolConfig(AbstractToolConfig['JITInstructionTool'], Generic[T]):
    """Configuration for JIT instruction tools."""

    type: Literal['jit'] = Field(default='jit', frozen=True)

    instruction: str = Field(...)

    def build(self) -> 'JITInstructionTool':
        raw_tool = GlobalToolCatalog().get_active_catalog().get_tool(self.name)
        return JITInstructionTool(
            raw_tool,
            jit_instruction=self.instruction,
        )


class JITInstructionTool(AbstractTool):
    """A tool decorator that provides just-in-time instructions after execution.

    Wraps another tool and appends a JIT (Just-In-Time) instruction to its output.
    This allows tools to provide dynamic guidance to the agent after execution,
    instructing how to interpret or act on the results.

    The instruction is added to the tool's output either as:
    - A dictionary with 'tool_result' and 'instruction' keys (for regular tools)
    - A final yielded value with 'instruction' key (for generator tools)

    This pattern enables:
    - Context-dependent post-execution guidance
    - Dynamic workflow steering based on tool results
    - Progressive disclosure of instructions
    - Tool-specific result interpretation hints

    Example JIT instructions:
    - "Use the search results to formulate your final answer"
    - "If the API returned an error, try the fallback endpoint"
    - "Format the data as JSON before responding"

    Note: This tool only supports synchronous functions and generators.

    Args:
        inner_tool: The tool to wrap with JIT instructions
        jit_instruction: Instruction text to append after tool execution
    """

    def __init__(self, inner_tool: AbstractTool, jit_instruction: str) -> None:
        raw = inner_tool.raw
        if inspect.iscoroutinefunction(raw) or inspect.isasyncgenfunction(raw):
            raise TypeError(
                'JITInstructionTool does not support async functions or async generators.'
            )

        self._inner = inner_tool
        self._jit_instruction = jit_instruction

        self.__instruction_field_name = 'instruction'

    @property
    def info(self) -> ToolInfo:
        return self._inner.info

    @property
    def raw(self) -> Callable[..., Any]:
        return self._inner.raw

    def clear_cache(self) -> None:
        return self._inner.clear_cache()

    def cache_info(self) -> Any:
        return self._inner.cache_info()

    def _wrap_generator(self, gen: Generator) -> Generator:
        for item in gen:
            yield item
        yield {self.__instruction_field_name: self._jit_instruction}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        tool_result = self._inner(*args, **kwargs)

        if isinstance(tool_result, GeneratorType):
            return self._wrap_generator(tool_result)

        return {
            'tool_result': tool_result,
            self.__instruction_field_name: self._jit_instruction,
        }

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    def __str__(self) -> str:
        base = str(self._inner)

        buf = StringIO()
        buf.write(base)
        buf.write(f'\tJIT Instruction: {self._jit_instruction}\n')

        return buf.getvalue()


@overload
def jit_tool(
    fn: Callable[[T], Any],
    *,
    jit_instruction: str,
    use_cache: bool = False,
    cache_size: int = 128,
) -> JITInstructionTool: ...


@overload
def jit_tool(
    fn: None = None,
    *,
    jit_instruction: str,
    use_cache: bool = False,
    cache_size: int = 128,
) -> Callable[[Callable[[T], Any]], JITInstructionTool]: ...


def jit_tool(
    fn: Callable[[T], Any] | None = None,
    *,
    jit_instruction: str,
    use_cache: bool = False,
    cache_size: int = 128,
) -> JITInstructionTool | Callable[[Callable[[T], Any]], JITInstructionTool]:
    def wrapper(f: Callable[[T], Any]) -> JITInstructionTool:
        raw_tool = Tool(f, use_cache=use_cache, cache_size=cache_size)
        return JITInstructionTool(raw_tool, jit_instruction=jit_instruction)

    if fn is not None:
        return wrapper(fn)

    return wrapper


@overload
def register_jit_tool(
    fn: Callable[[T], Any],
    *,
    jit_instruction: str,
    use_cache: bool = False,
    cache_size: int = 128,
    hidden: bool = False,
) -> JITInstructionTool: ...


@overload
def register_jit_tool(
    fn: None = None,
    *,
    jit_instruction: str,
    use_cache: bool = False,
    cache_size: int = 128,
    hidden: bool = False,
) -> Callable[[Callable[[T], Any]], JITInstructionTool]: ...


def register_jit_tool(
    fn: Callable[[T], Any] | None = None,
    *,
    jit_instruction: str,
    use_cache: bool = False,
    cache_size: int = 128,
    hidden: bool = False,
) -> JITInstructionTool | Callable[[Callable[[T], Any]], JITInstructionTool]:
    def wrapper(f: Callable[[T], Any]) -> JITInstructionTool:
        GlobalToolCatalog().get_active_catalog().register(
            f, use_cache=use_cache, cache_size=cache_size, hidden=hidden
        )
        return jit_tool(
            f,
            jit_instruction=jit_instruction,
            use_cache=use_cache,
            cache_size=cache_size,
        )

    if fn is not None:
        return wrapper(fn)

    return wrapper
