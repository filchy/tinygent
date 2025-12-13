from __future__ import annotations

from collections.abc import AsyncIterator
from io import StringIO
import os
import textwrap
import typing
from typing import Literal
from typing import override

from openai import AsyncOpenAI
from openai import OpenAI
from openai.lib.streaming.chat import ChunkEvent
from openai.types.chat import ChatCompletionFunctionToolParam

from tiny_openai.utils import openai_chunk_to_tiny_chunk
from tiny_openai.utils import openai_result_to_tiny_result
from tiny_openai.utils import tiny_prompt_to_openai_params
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm import AbstractLLMConfig
from tinygent.llms.utils import accumulate_llm_chunks
from tinygent.types.io.llm_io_chunks import TinyLLMResultChunk

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm import LLMStructuredT
    from tinygent.datamodels.tool import AbstractTool
    from tinygent.types.io.llm_io_input import TinyLLMInput
    from tinygent.types.io.llm_io_result import TinyLLMResult


class OpenAILLMConfig(AbstractLLMConfig['OpenAILLM']):
    type: Literal['openai'] = 'openai'

    model: str = 'gpt-4o'

    api_key: str | None = os.getenv('OPENAI_API_KEY', None)

    base_url: str | None = None

    temperature: float = 0.6

    timeout: float = 60.0

    def build(self) -> OpenAILLM:
        return OpenAILLM(
            model_name=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            timeout=self.timeout,
        )


class OpenAILLM(AbstractLLM[OpenAILLMConfig]):
    def __init__(
        self,
        model_name: str = 'gpt-4o',
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.6,
        timeout: float = 60.0,
    ) -> None:
        if not api_key and not (api_key := os.getenv('OPENAI_API_KEY', None)):
            raise ValueError(
                'OpenAI API key must be provided either via config',
                " or 'OPENAI_API_KEY' env variable.",
            )

        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout

        self.__sync_client: OpenAI | None = None
        self.__async_client: AsyncOpenAI | None = None

    @property
    def config(self) -> OpenAILLMConfig:
        return OpenAILLMConfig(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            timeout=self.timeout,
        )

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
                    'required': list(properties.keys()),
                    'additionalProperties': False,
                },
                'strict': True,
            },
        )

    def __get_sync_client(self) -> OpenAI:
        if self.__sync_client:
            return self.__sync_client

        self.__sync_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self.__sync_client

    def __get_async_client(self) -> AsyncOpenAI:
        if self.__async_client:
            return self.__async_client

        self.__async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self.__async_client

    def generate_text(
        self,
        llm_input: TinyLLMInput,
    ) -> TinyLLMResult:
        messages = tiny_prompt_to_openai_params(llm_input)

        res = self.__get_sync_client().chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            timeout=self.timeout,
        )

        return openai_result_to_tiny_result(res)

    async def agenerate_text(self, llm_input: TinyLLMInput) -> TinyLLMResult:
        messages = tiny_prompt_to_openai_params(llm_input)

        res = await self.__get_async_client().chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            timeout=self.timeout,
        )

        return openai_result_to_tiny_result(res)

    async def stream_text(
        self, llm_input: TinyLLMInput
    ) -> AsyncIterator[TinyLLMResultChunk]:
        messages = tiny_prompt_to_openai_params(llm_input)

        async with self.__get_async_client().chat.completions.stream(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            timeout=self.timeout,
        ) as stream:
            async for event in stream:
                if isinstance(event, ChunkEvent):
                    yield openai_chunk_to_tiny_chunk(event.chunk)

    def generate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT:
        messages = tiny_prompt_to_openai_params(llm_input)

        res = self.__get_sync_client().chat.completions.parse(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            timeout=self.timeout,
            response_format=output_schema,
        )

        if not (message := res.choices[0].message):
            raise ValueError('No message returned from OpenAI.')

        assert message.parsed is not None, 'Parsed response is None.'
        return message.parsed

    async def agenerate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT:
        messages = tiny_prompt_to_openai_params(llm_input)

        res = await self.__get_async_client().chat.completions.parse(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            timeout=self.timeout,
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
        messages = tiny_prompt_to_openai_params(llm_input)

        res = self.__get_sync_client().chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=functions,
            tool_choice='auto',
            temperature=self.temperature,
            timeout=self.timeout,
        )

        return openai_result_to_tiny_result(res)

    async def agenerate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        functions = [self._tool_convertor(tool) for tool in tools]
        messages = tiny_prompt_to_openai_params(llm_input)

        res = await self.__get_async_client().chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=functions,
            tool_choice='auto',
            temperature=self.temperature,
            timeout=self.timeout,
        )

        return openai_result_to_tiny_result(res)

    async def stream_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> AsyncIterator[TinyLLMResultChunk]:
        functions = [self._tool_convertor(tool) for tool in tools]
        messages = tiny_prompt_to_openai_params(llm_input)

        async with self.__get_async_client().chat.completions.stream(
            model=self.model_name,
            messages=messages,
            tools=functions,
            tool_choice='auto',
            temperature=self.temperature,
            timeout=self.timeout,
        ) as stream:

            async def tiny_chunks() -> AsyncIterator[TinyLLMResultChunk]:
                async for event in stream:
                    if isinstance(event, ChunkEvent):
                        yield openai_chunk_to_tiny_chunk(event.chunk)

            async for acc_chunk in accumulate_llm_chunks(tiny_chunks()):
                yield acc_chunk

    def __str__(self) -> str:
        buf = StringIO()

        buf.write('OpenAI LLM Summary:\n')
        buf.write(textwrap.indent(f'Model: {self.model_name}\n', '\t'))
        buf.write(textwrap.indent(f'Base URL: {self.base_url}\n', '\t'))
        buf.write(textwrap.indent(f'Temperature: {self.temperature}\n', '\t'))
        buf.write(textwrap.indent(f'Timeout: {self.timeout}\n', '\t'))

        return buf.getvalue()
