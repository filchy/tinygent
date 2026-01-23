from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncIterator
import typing
from typing import Generic
from typing import Iterable
from typing import TypeVar

from pydantic import Field
from pydantic import SecretStr

from tinygent.core.datamodels.messages import AllTinyMessages
from tinygent.core.types.base import TinyModel
from tinygent.core.types.builder import TinyModelBuildable

if typing.TYPE_CHECKING:
    from tinygent.core.datamodels.tool import AbstractTool
    from tinygent.core.types.io.llm_io_chunks import TinyLLMResultChunk
    from tinygent.core.types.io.llm_io_input import TinyLLMInput
    from tinygent.core.types.io.llm_io_result import TinyLLMResult

T = TypeVar('T', bound='AbstractLLM')
LLMConfigT = TypeVar('LLMConfigT', bound=TinyModel)
LLMStructuredT = TypeVar('LLMStructuredT', bound=TinyModel)


class AbstractLLMConfig(TinyModelBuildable[T], Generic[T]):
    """Abstract base class for LLM configurations."""

    model: str

    api_key: SecretStr | None = Field(default=None)

    timeout: float = Field(default=60.0)

    def build(self) -> T:
        """Build the LLM instance from the configuration."""
        raise NotImplementedError('Subclasses must implement this method.')


class AbstractLLM(ABC, Generic[LLMConfigT]):
    """Abstract base class for LLMs."""

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the LLM with the given configuration."""
        pass

    @property
    @abstractmethod
    def config(self) -> LLMConfigT:
        """Return the configuration of the LLM."""
        raise NotImplementedError('Subclasses must implement this method.')

    @property
    @abstractmethod
    def supports_tool_calls(self) -> bool:
        """Indicate whether the LLM supports tool calls."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def _tool_convertor(self, tool: AbstractTool) -> typing.Any:
        """Convert a tool to the format required by the LLM."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def generate_text(self, llm_input: TinyLLMInput) -> TinyLLMResult:
        """Generate text based on the given LLM input."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    async def agenerate_text(self, llm_input: TinyLLMInput) -> TinyLLMResult:
        """Asynchronously generate text based on the given LLM input."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def stream_text(self, llm_input: TinyLLMInput) -> AsyncIterator[TinyLLMResultChunk]:
        """Stream text generation based on the given LLM input."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def generate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT:
        """Generate structured data based on the given LLM input and output schema."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    async def agenerate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT:
        """Asynchronously generate structured data based on the given LLM input and output schema."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def generate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        """Generate text using the given LLM input and tools."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    async def agenerate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        """Asynchronously generate text using the given LLM input and tools."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def stream_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> AsyncIterator[TinyLLMResultChunk]:
        """Stream text generation using the given LLM input and tools."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def count_tokens_in_messages(self, messages: Iterable[AllTinyMessages]) -> int:
        """Count number of tokens from the message."""
        raise NotImplementedError('Subclasses must implement this method.')
