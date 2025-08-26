from __future__ import annotations

import typing

from abc import ABC
from abc import abstractmethod
from pydantic import BaseModel
from typing import Generic

if typing.TYPE_CHECKING:
    from langchain_core.prompt_values import PromptValue

    from tinygent.tools.tool import Tool
    from tinygent.datamodels.llm_result import TinyLLMResult

LLMConfigT = typing.TypeVar('LLMConfigT', bound=BaseModel)
LLMStructuredT = typing.TypeVar('LLMStructuredT', bound=BaseModel)


class AbstractLLM(ABC, Generic[LLMConfigT]):

    @abstractmethod
    def __init__(
        self,
        config: LLMConfigT,
        *args,
        **kwargs
    ) -> None: ...

    @property
    @abstractmethod
    def config(self) -> LLMConfigT: ...

    @property
    def supports_tool_calls(self) -> bool: ...

    @abstractmethod
    def _tool_convertor(
        self,
        tool: Tool
    ) -> typing.Any: ...

    @abstractmethod
    def generate_text(
        self,
        prompt: PromptValue
    ) -> TinyLLMResult: ...

    @abstractmethod
    async def agenerate_text(
        self,
        prompt: PromptValue
    ) -> TinyLLMResult: ...

    @abstractmethod
    def generate_structured(
        self,
        prompt: PromptValue,
        output_schema: LLMStructuredT
    ) -> LLMStructuredT: ...

    @abstractmethod
    async def agenerate_structured(
        self,
        prompt: PromptValue,
        output_schema: LLMStructuredT
    ) -> LLMStructuredT: ...

    @abstractmethod
    def generate_with_tools(
        self,
        prompt: PromptValue,
        tools: list[Tool]
    ) -> TinyLLMResult: ...

    @abstractmethod
    async def agenerate_with_tools(
        self,
        prompt: PromptValue,
        tools: list[Tool]
    ) -> TinyLLMResult: ...
