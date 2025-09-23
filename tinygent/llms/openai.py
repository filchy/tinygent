from __future__ import annotations

import os
import typing
from typing import override

from openai import AsyncOpenAI
from openai import OpenAI
from openai.types.chat import ChatCompletionFunctionToolParam
from pydantic import BaseModel

from tinygent.datamodels.llm import AbstractLLM
from tinygent.llms.utils import lc_prompt_to_openai_params
from tinygent.llms.utils import openai_result_to_tiny_result

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm import LLMStructuredT
    from tinygent.datamodels.llm_io import TinyLLMInput
    from tinygent.datamodels.llm_io import TinyLLMResult
    from tinygent.datamodels.tool import AbstractTool


class OpenAIConfig(BaseModel):
    model_name: str = 'gpt-4o'

    api_key: str | None = os.getenv('OPENAI_API_KEY', None)

    base_url: str | None = os.getenv('OPENAI_BASE_URL', None)

    temperature: float = 0.6

    timeout: float = 60.0


class OpenAILLM(AbstractLLM[OpenAIConfig]):
    def __init__(
        self,
        config: OpenAIConfig = OpenAIConfig(),
    ) -> None:
        if not config.api_key:
            raise ValueError(
                'OpenAI API key must be provided either via config',
                ' or \'OPENAI_API_KEY\' env variable.',
            )

        self._sync_client = OpenAI(api_key=config.api_key, base_url=config.base_url)

        self._async_client = AsyncOpenAI(
            api_key=config.api_key, base_url=config.base_url
        )

        self._config = config

    @property
    def config(self) -> OpenAIConfig:
        return self._config

    @property
    def supports_tool_calls(self) -> bool:
        return True

    @override
    def _tool_convertor(self, tool: AbstractTool) -> ChatCompletionFunctionToolParam:
        info = tool.info
        schema = info.input_schema

        def map_type(py_type: type) -> str:
            mapping = {
                str: 'string',
                int: 'integer',
                float: 'number',
                bool: 'boolean',
                list: 'array',
                dict: 'object',
            }
            return mapping.get(py_type, 'string')  # default fallback

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
                    'type': map_type(field_type),
                    'description': field.description,
                }

        return ChatCompletionFunctionToolParam(
            type='function',
            function={
                'name': info.name,
                'description': info.description,
                'parameters': {
                    'type': 'object',
                    'properties': properties,
                    'required': required,
                },
            },
        )

    def generate_text(
        self,
        llm_input: TinyLLMInput,
    ) -> TinyLLMResult:
        messages = lc_prompt_to_openai_params(llm_input)

        res = self._sync_client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self._config.temperature,
            timeout=self.config.timeout,
        )

        return openai_result_to_tiny_result(res)

    async def agenerate_text(self, llm_input: TinyLLMInput) -> TinyLLMResult:
        messages = lc_prompt_to_openai_params(llm_input)

        res = await self._async_client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self._config.temperature,
            timeout=self.config.timeout,
        )

        return openai_result_to_tiny_result(res)

    def generate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT:
        messages = lc_prompt_to_openai_params(llm_input)

        res = self._sync_client.chat.completions.parse(
            model=self.config.model_name,
            messages=messages,
            temperature=self._config.temperature,
            timeout=self.config.timeout,
            response_format=output_schema,
        )

        if not (message := res.choices[0].message):
            raise ValueError('No message returned from OpenAI.')

        assert message.parsed is not None, 'Parsed response is None.'
        return message.parsed

    async def agenerate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT:
        messages = lc_prompt_to_openai_params(llm_input)

        res = await self._async_client.chat.completions.parse(
            model=self.config.model_name,
            messages=messages,
            temperature=self._config.temperature,
            timeout=self.config.timeout,
            response_format=output_schema,
        )

        if not (message := res.choices[0].message):
            raise ValueError('No message returned from OpenAI.')

        assert message.parsed is not None, 'Parsed response is None.'
        return message.parsed

    def generate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        functions = [self._tool_convertor(tool) for tool in tools]
        messages = lc_prompt_to_openai_params(llm_input)

        res = self._sync_client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            tools=functions,
            tool_choice='auto',
            temperature=self._config.temperature,
            timeout=self.config.timeout,
        )

        return openai_result_to_tiny_result(res)

    async def agenerate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        functions = [self._tool_convertor(tool) for tool in tools]
        messages = lc_prompt_to_openai_params(llm_input)

        res = await self._async_client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            tools=functions,
            tool_choice='auto',
            temperature=self._config.temperature,
            timeout=self.config.timeout,
        )

        return openai_result_to_tiny_result(res)
