from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Generic
from typing import TypeVar

from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.types.builder import TinyModelBuildable
from tinygent.core.types.io.llm_io_input import TinyLLMInput

T = TypeVar('T', bound='AbstractMiddleware')


class AbstractMiddlewareConfig(TinyModelBuildable[T], Generic[T]):
    """Abstract base class for middleware configuration."""

    def build(self) -> T:
        """Build the middleware instance from the configuration."""
        raise NotImplementedError('Subclasses must implement this method.')


class AbstractMiddleware(ABC):
    """Abstract base class for middleware dispatcher."""

    @abstractmethod
    async def before_llm_call(
        self, *, run_id: str, llm_input: TinyLLMInput, kwargs: dict[str, Any]
    ) -> None:
        """Called before an LLM call is made."""
        pass

    @abstractmethod
    async def after_llm_call(
        self,
        *,
        run_id: str,
        llm_input: TinyLLMInput,
        result: Any,
        kwargs: dict[str, Any],
    ) -> None:
        """Called after an LLM call completes."""
        pass

    @abstractmethod
    async def before_tool_call(
        self,
        *,
        run_id: str,
        tool: AbstractTool,
        args: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> None:
        """Called before a tool is executed."""
        pass

    @abstractmethod
    async def after_tool_call(
        self,
        *,
        run_id: str,
        tool: AbstractTool,
        args: dict[str, Any],
        result: Any,
        kwargs: dict[str, Any],
    ) -> None:
        """Called after a tool execution completes."""
        pass

    @abstractmethod
    async def on_plan(self, *, run_id: str, plan: str, kwargs: dict[str, Any]) -> None:
        """Called when the agent creates a plan."""
        pass

    @abstractmethod
    async def on_reasoning(
        self, *, run_id: str, reasoning: str, kwargs: dict[str, Any]
    ) -> None:
        """Called when the agent produces a reasoning step."""
        pass

    @abstractmethod
    async def on_tool_reasoning(
        self, *, run_id: str, reasoning: str, kwargs: dict[str, Any]
    ) -> None:
        """Called when a tool produces reasoning output."""
        pass

    @abstractmethod
    async def on_answer(
        self, *, run_id: str, answer: str, kwargs: dict[str, Any]
    ) -> None:
        """Called when the agent produces a final answer."""
        pass

    @abstractmethod
    async def on_answer_chunk(
        self, *, run_id: str, chunk: str, idx: str, kwargs: dict[str, Any]
    ) -> None:
        """Called when the agent produces a chunk of the answer."""
        pass

    @abstractmethod
    async def on_error(
        self, *, run_id: str, e: Exception, kwargs: dict[str, Any]
    ) -> None:
        """Called when an error occurs during agent execution."""
        pass
