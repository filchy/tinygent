from dataclasses import replace
from io import StringIO
from typing import Any
from typing import Callable
from typing import Generic
from typing import Literal
from typing import TypeVar
from typing import cast
from typing import overload

from pydantic import Field
from pydantic import create_model

from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.datamodels.tool import AbstractToolConfig
from tinygent.core.runtime.tool_catalog import GlobalToolCatalog
from tinygent.core.types.base import TinyModel
from tinygent.tools.tool import Tool

T = TypeVar('T', bound=TinyModel)


class ReasoningToolConfig(AbstractToolConfig['ReasoningTool'], Generic[T]):
    """Configuration for reasoning tools."""

    type: Literal['reasoning'] = Field(default='reasoning', frozen=True)

    prompt: str = Field(...)

    def build(self) -> 'ReasoningTool':
        raw_tool = GlobalToolCatalog().get_active_catalog().get_tool(self.name)
        return ReasoningTool(
            raw_tool,
            reasoning_prompt=self.prompt,
        )


class ReasoningTool(AbstractTool):
    """A tool decorator that requires reasoning explanations for invocations.

    Wraps another tool and adds a mandatory "reasoning" parameter to its input
    schema. This forces the agent to explain why it's calling the tool before
    execution, promoting more deliberate tool use and providing insight into
    agent decision-making.

    The reasoning is captured and accessible via middleware hooks (on_tool_reasoning)
    but is not passed to the underlying tool function. This pattern is useful for:
    - Understanding agent tool selection logic
    - Debugging unexpected tool calls
    - Creating audit trails of agent reasoning
    - Training agents to think before acting

    The reasoning prompt can be customized to guide the type of explanation required
    (e.g., "Explain why this search is necessary" vs "Why this tool is being called").

    Args:
        inner_tool: The tool to wrap with reasoning requirements
        reasoning_prompt: Custom prompt for reasoning field (default: "Why this tool is being called")
    """

    def __init__(
        self, inner_tool: AbstractTool, reasoning_prompt: str | None = None
    ) -> None:
        self._inner = inner_tool
        self._reasoning: str | None = None
        self._reasoning_prompt = reasoning_prompt or 'Why this tool is being called'

        self.__reasoning_field_name = 'reasoning'

        # Dynamically create a new input schema with `reasoning: str`
        original_input = inner_tool.info.input_schema
        if original_input is None:
            raise TypeError('Tool must have an input schema')

        fields = {
            **{k: (v.annotation, v) for k, v in original_input.model_fields.items()},
            self.__reasoning_field_name: (
                str,
                Field(..., description=self._reasoning_prompt),
            ),
        }

        self._input_model = create_model(  # type: ignore[call-overload]
            f'{original_input.__name__}WithReasoning',
            __base__=original_input,
            **fields,
        )

    @property
    def reasoning(self) -> str:
        if self._reasoning is None:
            raise ValueError('Reasoning has not been set yet.')
        return self._reasoning

    @reasoning.setter
    def reasoning(self, _: str) -> None:
        raise ValueError('Reasoning is read-only and cannot be set directly.')

    @property
    def info(self) -> Any:
        inner_info = self._inner.info

        new_required_fields = [
            name
            for name, fld in self._input_model.model_fields.items()
            if fld.is_required()
        ]

        return replace(
            inner_info,
            input_schema=self._input_model,
            required_fields=new_required_fields,
        )

    @property
    def raw(self) -> Callable[..., Any]:
        return self._inner.raw

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], dict):
            data = self._input_model(**args[0])
        else:
            data = self._input_model(**kwargs)

        self._reasoning = getattr(cast(Any, data), self.__reasoning_field_name)

        orig_model = self._inner.info.input_schema
        assert orig_model is not None

        input_data = orig_model(
            **{
                k: v
                for k, v in data.model_dump().items()
                if k != self.__reasoning_field_name
            }
        )

        return self._inner(input_data)

    def clear_cache(self) -> None:
        return self._inner.clear_cache()

    def cache_info(self) -> Any:
        return self._inner.cache_info()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    def __str__(self) -> str:
        base = str(self._inner)

        buf = StringIO()
        buf.write(base)
        buf.write(f'\tReasoning Prompt: {self._reasoning_prompt}\n')

        return buf.getvalue()


@overload
def reasoning_tool(fn: Callable[[T], Any]) -> ReasoningTool: ...


@overload
def reasoning_tool(
    *,
    reasoning_prompt: str | None = None,
    use_cache: bool = False,
    cache_size: int = 128,
) -> Callable[[Callable[[T], Any]], ReasoningTool]: ...


def reasoning_tool(
    fn: Callable[[T], Any] | None = None,
    *,
    reasoning_prompt: str | None = None,
    use_cache: bool = False,
    cache_size: int = 128,
) -> ReasoningTool | Callable[[Callable[[T], Any]], ReasoningTool]:
    def wrapper(f: Callable[[T], Any]) -> ReasoningTool:
        raw_tool = Tool(f, use_cache=use_cache, cache_size=cache_size)
        return ReasoningTool(
            raw_tool,
            reasoning_prompt=reasoning_prompt,
        )

    if fn is None:
        return wrapper
    return wrapper(fn)


@overload
def register_reasoning_tool(fn: Callable[[T], Any]) -> ReasoningTool: ...


@overload
def register_reasoning_tool(
    *,
    reasoning_prompt: str | None = None,
    use_cache: bool = False,
    cache_size: int = 128,
    hidden: bool = False,
) -> Callable[[Callable[[T], Any]], ReasoningTool]: ...


def register_reasoning_tool(
    fn: Callable[[T], Any] | None = None,
    *,
    reasoning_prompt: str | None = None,
    use_cache: bool = False,
    cache_size: int = 128,
    hidden: bool = False,
) -> ReasoningTool | Callable[[Callable[[T], Any]], ReasoningTool]:
    def wrapper(f: Callable[[T], Any]) -> ReasoningTool:
        GlobalToolCatalog().get_active_catalog().register(
            f, use_cache=use_cache, cache_size=cache_size, hidden=hidden
        )
        raw_tool = Tool(f, use_cache=use_cache, cache_size=cache_size)
        return ReasoningTool(
            raw_tool,
            reasoning_prompt=reasoning_prompt,
        )

    if fn is None:
        return wrapper
    return wrapper(fn)
