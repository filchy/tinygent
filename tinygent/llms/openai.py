import os

from openai import AsyncOpenAI
from openai import OpenAI
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from typing import Literal
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import PromptValue

from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm import LLMStructuredT
from tinygent.tools.tool import Tool


class OpenAIConfig(BaseModel):

    api_key: str | None = os.getenv('OPENAI_API_KEY', None)

    base_url: str | None = None

    temperature: float = 0.6


class OpenAIFunction(BaseModel):

    _type: Literal['function'] = Field('function', alias='type')

    name: str

    description: str | None = None

    parameters: 'FunctionParams'

    class FunctionParams(BaseModel):

        _type: Literal['object'] = Field('object', alias='type')

        required: list[str] = []

        additional_properties: bool = False

        properties: dict[str, 'Property'] = {}

        class Property(BaseModel):

            _type: str = Field(..., alias='type')

            description: str | None = None

        @model_validator(mode='after')
        def check_required(self) -> 'OpenAIFunction.FunctionParams':

            missing = [
                key for key in self.required
                if key not in self.properties
            ]

            if missing:
                raise ValueError(
                    f'Required properties missing: {missing}.'
                )
            return self


class OpenAILLM(AbstractLLM[OpenAIConfig]):

    def __init__(
        self,
        config: OpenAIConfig
    ) -> None:

        if not config.api_key:
            raise ValueError(
                'OpenAI API key must be provided either via config',
                ' or \'OPENAI_API_KEY\' env variable.'
            )

        self._sync_client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

        self._async_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

        self._config = config

    @property
    def config(self) -> OpenAIConfig:

        return self._config

    @property
    def supports_tool_calls(self) -> bool:

        return True

    def generate_text(
        self,
        prompt: PromptValue,
    ) -> LLMResult:

        raise NotImplementedError()

    async def agenerate_text(
        self,
        prompt: PromptValue
    ) -> LLMResult:

        raise NotImplementedError()

    def generate_structured(
        self,
        prompt: PromptValue,
        output_schema: LLMStructuredT
    ) -> LLMStructuredT:

        raise NotImplementedError()

    async def agenerate_structured(
        self,
        prompt: PromptValue,
        output_schema: LLMStructuredT
    ) -> LLMStructuredT:

        raise NotImplementedError()

    def generate_with_tool(
        self,
        prompt: PromptValue,
        tools: list[Tool]
    ) -> LLMResult:

        raise NotImplementedError()

    async def agenerate_with_tool(
        self,
        prompt: PromptValue,
        tools: list[Tool]
    ) -> LLMResult:

        raise NotImplementedError()
