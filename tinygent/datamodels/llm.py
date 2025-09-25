from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import typing
from typing import Generic

from pydantic import BaseModel

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm_io import TinyLLMInput
    from tinygent.datamodels.llm_io import TinyLLMResult
    from tinygent.datamodels.tool import AbstractTool

LLMConfigT = typing.TypeVar('LLMConfigT', bound=BaseModel)
LLMStructuredT = typing.TypeVar('LLMStructuredT', bound=BaseModel)


class AbstractLLM(ABC, Generic[LLMConfigT]):
    """Abstract base class for LLMs."""

    @abstractmethod
    def __init__(self, config: LLMConfigT, *args, **kwargs) -> None: ...
    """Initialize the LLM with the given configuration."""

    @property
    @abstractmethod
    def config(self) -> LLMConfigT: ...
    """Get the configuration of the LLM."""

    @property
    @abstractmethod
    def supports_tool_calls(self) -> bool: ...
    """Indicate whether the LLM supports tool calls."""

    @abstractmethod
    def _tool_convertor(self, tool: AbstractTool) -> typing.Any: ...
    """Convert a tool to the format required by the LLM."""

    @abstractmethod
    def generate_text(self, llm_input: TinyLLMInput) -> TinyLLMResult: ...
    """Generate text based on the given LLM input."""

    @abstractmethod
    async def agenerate_text(self, llm_input: TinyLLMInput) -> TinyLLMResult: ...
    """Asynchronously generate text based on the given LLM input."""

    @abstractmethod
    def generate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT: ...
    """Generate structured data based on the given LLM input and output schema."""

    @abstractmethod
    async def agenerate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT: ...
    """Asynchronously generate structured data based on the given LLM input and output schema."""

    @abstractmethod
    def generate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult: ...
    """Generate text using the given LLM input and tools."""

    @abstractmethod
    async def agenerate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult: ...
    """Asynchronously generate text using the given LLM input and tools."""
