from __future__ import annotations

from collections.abc import AsyncIterator
from io import StringIO
import json
import os
import textwrap
import typing
from typing import Literal
from typing import override

from mistralai import Function
from mistralai import Mistral
from mistralai import Tool

from tiny_mistralai.utils import mistralai_chunk_to_tiny_chunks
from tiny_mistralai.utils import mistralai_result_to_tiny_result
from tiny_mistralai.utils import tiny_prompt_to_mistralai_params
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm import AbstractLLMConfig
from tinygent.datamodels.llm_io_chunks import TinyLLMResultChunk
from tinygent.llms.utils import accumulate_llm_chunks

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm import LLMStructuredT
    from tinygent.datamodels.llm_io_input import TinyLLMInput
    from tinygent.datamodels.llm_io_result import TinyLLMResult
    from tinygent.datamodels.tool import AbstractTool


class MistralAIConfig(AbstractLLMConfig['MistralAILLM']):
    type: Literal['mistralai'] = 'mistralai'

    model: str = 'mistral-medium-latest'

    api_key: str | None = os.getenv('MISTRALAI_API_KEY', None)

    safe_prompt: bool = True

    temperature: float = 0.6

    timeout: float = 60.0

    def build(self) -> MistralAILLM:
        return MistralAILLM(
            model_name=self.model,
            api_key=self.api_key,
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout=self.timeout,
        )


class MistralAILLM(AbstractLLM[MistralAIConfig]):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        safe_prompt: bool = True,
        temperature: float = 0.6,
        timeout: float = 60.0,
    ) -> None:
        if not api_key and not (api_key := os.getenv('MISTRALAI_API_KEY', None)):
            raise ValueError(
                'MistralAI API key must be provided either via config'
                "or 'MISTRALAI_API_KEY' env variable."
            )

        self._client = Mistral(api_key=api_key)

        self.model_name = model_name
        self.api_key = api_key
        self.safe_prompt = safe_prompt
        self.temperature = temperature
        self.timeout = timeout

    @property
    def config(self) -> MistralAIConfig:
        return MistralAIConfig(
            model=self.model_name,
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout=self.timeout,
        )

    @property
    def supports_tool_calls(self) -> bool:
        return True  # INFO: Not all models may support tool calls, but mistralai api error if not.

    @override
    def _tool_convertor(self, tool: AbstractTool) -> Tool:
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

        return Tool(
            type='function',
            function=Function(
                name=info.name,
                description=info.description,
                parameters={
                    'type': 'object',
                    'properties': properties,
                    'required': info.required_fields,
                },
            ),
        )

    def generate_text(
        self,
        llm_input: TinyLLMInput,
    ) -> TinyLLMResult:
        messages = tiny_prompt_to_mistralai_params(llm_input)

        res = self._client.chat.complete(
            model=self.model_name,
            messages=messages,
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout_ms=int(self.timeout * 1000),
        )

        return mistralai_result_to_tiny_result(res)

    async def agenerate_text(
        self,
        llm_input: TinyLLMInput,
    ) -> TinyLLMResult:
        messages = tiny_prompt_to_mistralai_params(llm_input)

        res = await self._client.chat.complete_async(
            model=self.model_name,
            messages=messages,
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout_ms=int(self.timeout * 1000),
        )

        return mistralai_result_to_tiny_result(res)

    async def stream_text(
        self, llm_input: TinyLLMInput
    ) -> AsyncIterator[TinyLLMResultChunk]:
        messages = tiny_prompt_to_mistralai_params(llm_input)

        res = await self._client.chat.stream_async(
            model=self.model_name,
            messages=messages,
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout_ms=int(self.timeout * 1000),
        )

        async for chunk in res:
            for tiny_chunk in mistralai_chunk_to_tiny_chunks(chunk.data):
                yield tiny_chunk

    def generate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT:
        messages = tiny_prompt_to_mistralai_params(llm_input)

        res = self._client.chat.parse(
            model=self.model_name,
            messages=messages,
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout_ms=int(self.timeout * 1000),
            response_format=output_schema,
        )

        if not res.choices or not (message := res.choices[0].message):
            raise ValueError('No message in MistralAI response.')

        return output_schema.model_validate(json.loads(str(message.content) or '{}'))

    async def agenerate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT:
        messages = tiny_prompt_to_mistralai_params(llm_input)

        res = await self._client.chat.parse_async(
            model=self.model_name,
            messages=messages,
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout_ms=int(self.timeout * 1000),
            response_format=output_schema,
        )

        if not res.choices or not (message := res.choices[0].message):
            raise ValueError('No message in MistralAI response.')

        return output_schema.model_validate(json.loads(str(message.content) or '{}'))

    def generate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        functions = [self._tool_convertor(tool) for tool in tools]
        messages = tiny_prompt_to_mistralai_params(llm_input)

        res = self._client.chat.complete(
            model=self.model_name,
            messages=messages,
            tools=functions,
            tool_choice='auto',
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout_ms=int(self.timeout * 1000),
        )

        return mistralai_result_to_tiny_result(res)

    async def agenerate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        functions = [self._tool_convertor(tool) for tool in tools]
        messages = tiny_prompt_to_mistralai_params(llm_input)

        res = await self._client.chat.complete_async(
            model=self.model_name,
            messages=messages,
            tools=functions,
            tool_choice='auto',
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout_ms=int(self.timeout * 1000),
        )

        return mistralai_result_to_tiny_result(res)

    async def stream_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> AsyncIterator[TinyLLMResultChunk]:
        functions = [self._tool_convertor(tool) for tool in tools]
        messages = tiny_prompt_to_mistralai_params(llm_input)

        res = await self._client.chat.stream_async(
            model=self.model_name,
            messages=messages,
            tools=functions,
            tool_choice='auto',
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout_ms=int(self.timeout * 1000),
        )

        async def raw_chunks() -> AsyncIterator[TinyLLMResultChunk]:
            async for chunk in res:
                for tiny_chunk in mistralai_chunk_to_tiny_chunks(chunk.data):
                    yield tiny_chunk

        async for acc_chunk in accumulate_llm_chunks(raw_chunks()):
            yield acc_chunk

    def __str__(self) -> str:
        buf = StringIO()

        buf.write('OpenAI LLM Summary:\n')
        buf.write(textwrap.indent(f'Model: {self.model_name}\n', '\t'))
        buf.write(textwrap.indent(f'Safe Prompt: {self.safe_prompt}\n', '\t'))
        buf.write(textwrap.indent(f'Temperature: {self.temperature}\n', '\t'))
        buf.write(textwrap.indent(f'Timeout: {self.timeout}\n', '\t'))

        return buf.getvalue()
