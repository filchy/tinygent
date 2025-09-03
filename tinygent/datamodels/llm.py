from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import typing
from typing import Generic

from pydantic import BaseModel

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm_io import TinyLLMInput
    from tinygent.datamodels.llm_io import TinyLLMResult
    from tinygent.tools.tool import Tool

LLMConfigT = typing.TypeVar('LLMConfigT', bound=BaseModel)
LLMStructuredT = typing.TypeVar('LLMStructuredT', bound=BaseModel)


class AbstractLLM(ABC, Generic[LLMConfigT]):
    @abstractmethod
    def __init__(self, config: LLMConfigT, *args, **kwargs) -> None: ...

    @property
    @abstractmethod
    def config(self) -> LLMConfigT: ...

    @property
    @abstractmethod
    def supports_tool_calls(self) -> bool: ...

    @abstractmethod
    def _tool_convertor(self, tool: Tool) -> typing.Any: ...

    @abstractmethod
    def generate_text(self, prompt: TinyLLMInput) -> TinyLLMResult: ...

    @abstractmethod
    async def agenerate_text(self, prompt: TinyLLMInput) -> TinyLLMResult: ...

    @abstractmethod
    def generate_structured(
        self, prompt: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT: ...

    @abstractmethod
    async def agenerate_structured(
        self, prompt: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT: ...

    @abstractmethod
    def generate_with_tools(
        self, prompt: TinyLLMInput, tools: list[Tool]
    ) -> TinyLLMResult: ...

    @abstractmethod
    async def agenerate_with_tools(
        self, prompt: TinyLLMInput, tools: list[Tool]
    ) -> TinyLLMResult: ...
