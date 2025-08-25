import typing

from abc import ABC
from abc import abstractmethod
from langchain_core.prompt_values import PromptValue
from langchain_core.outputs import LLMResult
from pydantic import BaseModel
from typing import Generic

from tinygent.tools.tool import Tool

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
    ) -> LLMResult: ...

    @abstractmethod
    async def agenerate_text(
        self,
        prompt: PromptValue
    ) -> LLMResult: ...

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
        tools: list['Tool']
    ) -> LLMResult: ...

    @abstractmethod
    async def agenerate_with_tools(
        self,
        prompt: PromptValue,
        tools: list['Tool']
    ) -> LLMResult: ...
