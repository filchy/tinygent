import os

from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import PromptValue
from openai import AsyncOpenAI
from openai import OpenAI
from openai.types.chat import ChatCompletionFunctionToolParam
from pydantic import BaseModel
from pydantic import model_validator
from typing import Literal

from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm import LLMStructuredT
from tinygent.llms.converters.openai import lc_prompt_to_openai_params
from tinygent.llms.converters.openai import openai_result_to_lc_result
from tinygent.tools.tool import Tool


class OpenAIConfig(BaseModel):

    model_name: str = 'gpt-4o'

    api_key: str | None = os.getenv('OPENAI_API_KEY', None)

    base_url: str | None = os.getenv('OPENAI_BASE_URL', None)

    temperature: float = 0.6


class OpenAIFunction(BaseModel):

    type: Literal['function'] = 'function'

    name: str

    description: str | None = None

    parameters: 'FunctionParams'

    class FunctionParams(BaseModel):

        type: Literal['object'] = 'object'

        required: list[str] = []

        additional_properties: bool = False

        properties: dict[str, 'Property'] = {}

        class Property(BaseModel):

            type: str

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
        config: OpenAIConfig = OpenAIConfig(),
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

    def _tool_convertor(self, tool: Tool) -> ChatCompletionFunctionToolParam:
        info = tool.info
        schema = info.input_schema

        def map_type(py_type: type) -> str:
            mapping = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
            }
            return mapping.get(py_type, "string")  # default fallback

        properties = {}
        required = info.required_fields

        if schema:
            for name, field in schema.model_fields.items():
                field_type = (
                    field.annotation
                    if isinstance(field.annotation, type)
                    else type(field.annotation)
                )
                properties[name] = {
                    "type": map_type(field_type),
                    "description": field.description,
                }

        return ChatCompletionFunctionToolParam(
            type="function",
            function={
                "name": info.name,
                "description": info.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        )

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

    def generate_with_tools(
        self,
        prompt: PromptValue,
        tools: list[Tool]
    ) -> LLMResult:

        functions = [self._tool_convertor(tool) for tool in tools]

        messages = lc_prompt_to_openai_params(prompt)

        res = self._sync_client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            tools=functions,
            tool_choice='auto',
            temperature=self._config.temperature
        )

        return openai_result_to_lc_result(res)

    async def agenerate_with_tools(
        self,
        prompt: PromptValue,
        tools: list[Tool]
    ) -> LLMResult:

        raise NotImplementedError()
