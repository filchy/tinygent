from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import typing
from typing import Generic

from tinygent.types.base import TinyModel

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm_io import TinyLLMInput
    from tinygent.datamodels.llm_io import TinyLLMResult
    from tinygent.datamodels.tool import AbstractTool

LLMConfigT = typing.TypeVar('LLMConfigT', bound=TinyModel)
LLMStructuredT = typing.TypeVar('LLMStructuredT', bound=TinyModel)


class AbstractLLM(ABC, Generic[LLMConfigT]):
    """Abstract base class for LLMs."""

    @abstractmethod
    def __init__(self, config: LLMConfigT, *args, **kwargs) -> None:
        """Initialize the LLM with the given configuration."""
        pass

    @property
    @abstractmethod
    def config(self) -> LLMConfigT:
        """Get the configuration of the LLM."""
        pass

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
