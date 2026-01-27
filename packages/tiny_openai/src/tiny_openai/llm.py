from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache
from io import StringIO
import os
import textwrap
import typing
from typing import Iterable
from typing import Literal
from typing import override

from openai import AsyncOpenAI
from openai import OpenAI
from openai.lib.streaming.chat import ChunkEvent
from openai.types.chat import ChatCompletionFunctionToolParam
from pydantic import Field
from pydantic import SecretStr

from tiny_openai.utils import openai_chunk_to_tiny_chunk
from tiny_openai.utils import openai_result_to_tiny_result
from tiny_openai.utils import tiny_prompt_to_openai_params
from tinygent.core.datamodels.llm import AbstractLLM
from tinygent.core.datamodels.llm import AbstractLLMConfig
from tinygent.core.datamodels.messages import AllTinyMessages
from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.otel import set_tiny_attribute
from tinygent.core.telemetry.utils import set_llm_telemetry_attributes
from tinygent.core.types.io.llm_io_chunks import TinyLLMResultChunk
from tinygent.llms.utils import accumulate_llm_chunks
from tinygent.llms.utils import group_chunks_for_telemetry

if typing.TYPE_CHECKING:
    import tiktoken

    from tinygent.core.datamodels.llm import LLMStructuredT
    from tinygent.core.datamodels.tool import AbstractTool
    from tinygent.core.types.io.llm_io_input import TinyLLMInput
    from tinygent.core.types.io.llm_io_result import TinyLLMResult


class OpenAILLMConfig(AbstractLLMConfig['OpenAILLM']):
    type: Literal['openai'] = Field(default='openai', frozen=True)

    model: str = Field(default='gpt-4o')

    api_key: SecretStr | None = Field(
        default_factory=lambda: (
            SecretStr(os.environ['OPENAI_API_KEY'])
            if 'OPENAI_API_KEY' in os.environ
            else None
        ),
    )

    base_url: str | None = Field(default=None)

    temperature: float | None = Field(default=None)

    timeout: float = Field(default=60.0)

    def build(self) -> OpenAILLM:
        return OpenAILLM(
            model=self.model,
            api_key=self.api_key.get_secret_value() if self.api_key else None,
            base_url=self.base_url,
            temperature=self.temperature,
            timeout=self.timeout,
        )


class OpenAILLM(AbstractLLM[OpenAILLMConfig]):
    def __init__(
        self,
        model: str = 'gpt-4o',
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key and not (api_key := os.getenv('OPENAI_API_KEY', None)):
            raise ValueError(
                'OpenAI API key must be provided either via config',
                " or 'OPENAI_API_KEY' env variable.",
            )

        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout

        self.__sync_client: OpenAI | None = None
        self.__async_client: AsyncOpenAI | None = None

    @property
    def config(self) -> OpenAILLMConfig:
        return OpenAILLMConfig(
            model=self.model,
            api_key=SecretStr(self.api_key),
            base_url=self.base_url,
            temperature=self.temperature,
            timeout=self.timeout,
        )

    @property
    def supports_tool_calls(self) -> bool:
        return True

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_encoding(model: str) -> tiktoken.Encoding:
        import tiktoken

        return tiktoken.encoding_for_model(model)

    @staticmethod
    @lru_cache(maxsize=100_000)
    def _count_tokens(model: str, text: str) -> int:
        return len(OpenAILLM._get_encoding(model).encode(text))

    def _request_args(self) -> dict:
        args = {
            'model': self.model,
            'timeout': self.timeout,
        }
        if self.temperature:
            args['temperature'] = self.temperature
        return args

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

                prop = {'type': map_type(field_type)}
                if field.description:
                    prop['description'] = field.description

                properties[name] = prop

        return ChatCompletionFunctionToolParam(
            type='function',
            function={
                'name': info.name,
                'description': info.description,
                'parameters': {
                    'type': 'object',
                    'properties': properties,
                    'required': info.required_fields,
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

    @tiny_trace('generate_text')
    def generate_text(
        self,
        llm_input: TinyLLMInput,
    ) -> TinyLLMResult:
        messages = tiny_prompt_to_openai_params(llm_input)

        res = self.__get_sync_client().chat.completions.create(
            messages=messages,
            **self._request_args(),
        )

        tiny_res = openai_result_to_tiny_result(res)
        set_llm_telemetry_attributes(
            self.config, llm_input.messages, result=tiny_res.to_string()
        )
        return tiny_res

    @tiny_trace('agenerate_text')
    async def agenerate_text(self, llm_input: TinyLLMInput) -> TinyLLMResult:
        messages = tiny_prompt_to_openai_params(llm_input)

        res = await self.__get_async_client().chat.completions.create(
            messages=messages,
            **self._request_args(),
        )

        tiny_res = openai_result_to_tiny_result(res)
        set_llm_telemetry_attributes(
            self.config, llm_input.messages, result=tiny_res.to_string()
        )
        return tiny_res

    @tiny_trace('stream_text')
    async def stream_text(
        self, llm_input: TinyLLMInput
    ) -> AsyncIterator[TinyLLMResultChunk]:
        messages = tiny_prompt_to_openai_params(llm_input)
        set_llm_telemetry_attributes(self.config, llm_input.messages)

        async with self.__get_async_client().chat.completions.stream(
            messages=messages,
            **self._request_args(),
        ) as stream:

            async def tiny_chunks() -> AsyncIterator[TinyLLMResultChunk]:
                async for event in stream:
                    if isinstance(event, ChunkEvent):
                        yield openai_chunk_to_tiny_chunk(event.chunk)

            accumulated_chunks: list[TinyLLMResultChunk] = []
            try:
                async for acc_chunk in accumulate_llm_chunks(tiny_chunks()):
                    accumulated_chunks.append(acc_chunk)
                    yield acc_chunk
            finally:
                set_tiny_attribute(
                    'result',
                    group_chunks_for_telemetry(accumulated_chunks),
                )

    @tiny_trace('generate_structured')
    def generate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT:
        messages = tiny_prompt_to_openai_params(llm_input)

        res = self.__get_sync_client().chat.completions.parse(
            messages=messages,
            response_format=output_schema,
            **self._request_args(),
        )

        if not (message := res.choices[0].message):
            raise ValueError('No message returned from OpenAI.')

        assert message.parsed is not None, 'Parsed response is None.'

        set_llm_telemetry_attributes(
            self.config,
            llm_input.messages,
            result=str(message.parsed),
            output_schema=output_schema,
        )
        return message.parsed

    @tiny_trace('agenerate_structured')
    async def agenerate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT:
        messages = tiny_prompt_to_openai_params(llm_input)

        res = await self.__get_async_client().chat.completions.parse(
            messages=messages, response_format=output_schema, **self._request_args()
        )

        if not (message := res.choices[0].message):
            raise ValueError('No message returned from OpenAI.')

        assert message.parsed is not None, 'Parsed response is None.'

        set_llm_telemetry_attributes(
            self.config,
            llm_input.messages,
            result=str(message.parsed),
            output_schema=output_schema,
        )
        return message.parsed

    @tiny_trace('generate_with_tools')
    def generate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        functions = [self._tool_convertor(tool) for tool in tools]
        messages = tiny_prompt_to_openai_params(llm_input)

        res = self.__get_sync_client().chat.completions.create(
            messages=messages,
            tools=functions,
            tool_choice='auto',
            **self._request_args(),
        )

        tiny_res = openai_result_to_tiny_result(res)
        set_llm_telemetry_attributes(
            self.config, llm_input.messages, result=tiny_res.to_string(), tools=tools
        )
        return tiny_res

    @tiny_trace('agenerate_with_tools')
    async def agenerate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        functions = [self._tool_convertor(tool) for tool in tools]
        messages = tiny_prompt_to_openai_params(llm_input)

        res = await self.__get_async_client().chat.completions.create(
            messages=messages,
            tools=functions,
            tool_choice='auto',
            **self._request_args(),
        )

        tiny_res = openai_result_to_tiny_result(res)
        set_llm_telemetry_attributes(
            self.config, llm_input.messages, result=tiny_res.to_string(), tools=tools
        )
        return tiny_res

    @tiny_trace('stream_with_tools')
    async def stream_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> AsyncIterator[TinyLLMResultChunk]:
        functions = [self._tool_convertor(tool) for tool in tools]
        messages = tiny_prompt_to_openai_params(llm_input)
        set_llm_telemetry_attributes(self.config, llm_input.messages, tools=tools)

        async with self.__get_async_client().chat.completions.stream(
            messages=messages,
            tools=functions,
            tool_choice='auto',
            **self._request_args(),
        ) as stream:

            async def tiny_chunks() -> AsyncIterator[TinyLLMResultChunk]:
                async for event in stream:
                    if isinstance(event, ChunkEvent):
                        yield openai_chunk_to_tiny_chunk(event.chunk)

            accumulated_chunks: list[TinyLLMResultChunk] = []
            try:
                async for acc_chunk in accumulate_llm_chunks(tiny_chunks()):
                    accumulated_chunks.append(acc_chunk)
                    yield acc_chunk
            finally:
                set_tiny_attribute(
                    'result',
                    group_chunks_for_telemetry(accumulated_chunks),
                )

    @tiny_trace('count_tokens_in_messages')
    def count_tokens_in_messages(self, messages: Iterable[AllTinyMessages]) -> int:
        set_llm_telemetry_attributes(self.config, messages)

        number_of_tokens = sum(
            [OpenAILLM._count_tokens(self.model, m.tiny_str) for m in messages]
        )

        set_tiny_attribute('number_of_tokens', number_of_tokens)
        return number_of_tokens

    def __str__(self) -> str:
        buf = StringIO()

        buf.write('OpenAI LLM Summary:\n')
        buf.write(textwrap.indent(f'Model: {self.model}\n', '\t'))
        buf.write(textwrap.indent(f'Base URL: {self.base_url}\n', '\t'))
        buf.write(textwrap.indent(f'Temperature: {self.temperature}\n', '\t'))
        buf.write(textwrap.indent(f'Timeout: {self.timeout}\n', '\t'))

        return buf.getvalue()
