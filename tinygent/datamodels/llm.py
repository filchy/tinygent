import typing

from abc import ABC
from abc import abstractmethod
from langchain_core.outputs import LLMResult
from pydantic import BaseModel
from typing import Generic

if typing.TYPE_CHECKING:
    from langchain_core.prompt_values import PromptValue

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
