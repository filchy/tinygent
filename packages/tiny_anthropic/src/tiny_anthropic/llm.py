from __future__ import annotations

from collections.abc import AsyncIterator
import os
import textwrap
import typing
from typing import Iterable
from typing import Literal
from typing import override

from anthropic import Anthropic
from anthropic import AsyncAnthropic
from anthropic.types import ToolParam
from pydantic import Field
from pydantic import SecretStr

from tiny_anthropic.utils import anthropic_chunk_to_tiny_chunk
from tiny_anthropic.utils import anthropic_result_to_tiny_result
from tiny_anthropic.utils import tiny_prompt_to_anthropic_params
from tinygent.core.datamodels.llm import AbstractLLM
from tinygent.core.datamodels.llm import AbstractLLMConfig
from tinygent.core.datamodels.messages import AllTinyMessages
from tinygent.core.datamodels.messages import TinyToolCall
from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.otel import set_tiny_attribute
from tinygent.core.telemetry.utils import set_llm_telemetry_attributes
from tinygent.core.types.io.llm_io_chunks import TinyLLMResultChunk
from tinygent.core.types.io.llm_io_input import TinyLLMInput
from tinygent.llms.utils import StringIO
from tinygent.llms.utils import accumulate_llm_chunks
from tinygent.llms.utils import group_chunks_for_telemetry

if typing.TYPE_CHECKING:
    from tinygent.core.datamodels.llm import LLMStructuredT
    from tinygent.core.datamodels.tool import AbstractTool
    from tinygent.core.types.io.llm_io_result import TinyLLMResult

_anthropic_sturcured_outputs_beta = 'structured-outputs-2025-11-13'
_anthropic_tool_use_beta = 'advanced-tool-use-2025-11-20'
_anthropic_tool_streaming_beta = 'fine-grained-tool-streaming-2025-05-14'


class ClaudeLLMConfig(AbstractLLMConfig['ClaudeLLM']):
    type: Literal['anthropic'] = Field(default='anthropic', frozen=True)

    model: str = Field(default='claude-sonnet-4-5')

    api_key: SecretStr | None = Field(
        default_factory=lambda: (
            SecretStr(os.environ['CLAUDE_API_KEY'])
            if 'CLAUDE_API_KEY' in os.environ
            else None
        ),
    )

    base_url: str | None = Field(default=None)

    max_tokens: int = Field(default=1000)

    timeout: float = Field(default=60.0)

    def build(self) -> ClaudeLLM:
        return ClaudeLLM(
            model=self.model,
            api_key=self.api_key.get_secret_value() if self.api_key else None,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )


class ClaudeLLM(AbstractLLM[ClaudeLLMConfig]):
    def __init__(
        self,
        model: str = 'claude-sonnet-4-5',
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 1000,
        timeout: float = 60.0,
    ) -> None:
        if not api_key and not (api_key := os.getenv('CLAUDE_API_KEY', None)):
            raise ValueError(
                'Claude API key must be provided either via config',
                " or 'CLAUDE_API_KEY' env variable.",
            )

        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout = timeout

        self.__sync_client: Anthropic | None = None
        self.__async_client: AsyncAnthropic | None = None

    @property
    def config(self) -> ClaudeLLMConfig:
        return ClaudeLLMConfig(
            model=self.model,
            api_key=SecretStr(self.api_key),
            base_url=self.base_url,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
        )

    @property
    def supports_tool_calls(self) -> bool:
        return True

    def __get_sync_client(self) -> Anthropic:
        if self.__sync_client:
            return self.__sync_client

        self.__sync_client = Anthropic(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )
        return self.__sync_client

    def __get_async_client(self) -> AsyncAnthropic:
        if self.__async_client:
            return self.__async_client

        self.__async_client = AsyncAnthropic(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )
        return self.__async_client

    @override
    def _tool_convertor(self, tool: AbstractTool) -> ToolParam:
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

        properties: dict[str, object] = {}

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

        return {
            'name': info.name,
            'description': info.description,
            'input_schema': {
                'type': 'object',
                'properties': properties,
                'required': info.required_fields,
            },
        }

    def __create_client_kwargs(self, llm_input: TinyLLMInput) -> dict:
        messages, system = tiny_prompt_to_anthropic_params(llm_input)

        kwargs = {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'messages': messages,
        }
        if system:
            kwargs['system'] = system
        return kwargs

    @tiny_trace('generate_text')
    def generate_text(self, llm_input: TinyLLMInput) -> TinyLLMResult:
        kwargs = self.__create_client_kwargs(llm_input)

        res = self.__get_sync_client().messages.create(**kwargs)

        tiny_res = anthropic_result_to_tiny_result(res)
        set_llm_telemetry_attributes(
            self.config, llm_input.messages, result=tiny_res.to_string()
        )
        return tiny_res

    @tiny_trace('agenerate_text')
    async def agenerate_text(self, llm_input: TinyLLMInput) -> TinyLLMResult:
        kwargs = self.__create_client_kwargs(llm_input)

        res = await self.__get_async_client().messages.create(**kwargs)

        tiny_res = anthropic_result_to_tiny_result(res)
        set_llm_telemetry_attributes(
            self.config, llm_input.messages, result=tiny_res.to_string()
        )
        return tiny_res

    @tiny_trace('stream_text')
    async def stream_text(
        self, llm_input: TinyLLMInput
    ) -> AsyncIterator[TinyLLMResultChunk]:
        kwargs = self.__create_client_kwargs(llm_input)
        set_llm_telemetry_attributes(self.config, llm_input.messages)

        async def tiny_chunks() -> AsyncIterator[TinyLLMResultChunk]:
            async with self.__get_async_client().messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield anthropic_chunk_to_tiny_chunk(text)

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
        kwargs = self.__create_client_kwargs(llm_input)
        kwargs['output_format'] = output_schema
        kwargs['betas'] = [_anthropic_sturcured_outputs_beta]

        res = self.__get_sync_client().beta.messages.parse(**kwargs)
        if not (p := res.parsed_output):
            raise ValueError('Parsed response is None.')

        set_llm_telemetry_attributes(
            self.config,
            llm_input.messages,
            result=str(p),
            output_schema=output_schema,
        )
        return p

    @tiny_trace('agenerate_structured')
    async def agenerate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT:
        kwargs = self.__create_client_kwargs(llm_input)
        kwargs['output_format'] = output_schema
        kwargs['betas'] = [_anthropic_sturcured_outputs_beta]

        res = await self.__get_async_client().beta.messages.parse(**kwargs)
        if not (p := res.parsed_output):
            raise ValueError('Parsed response is None.')

        set_llm_telemetry_attributes(
            self.config,
            llm_input.messages,
            result=str(p),
            output_schema=output_schema,
        )
        return p

    @tiny_trace('generate_with_tools')
    def generate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        kwargs = self.__create_client_kwargs(llm_input)
        kwargs['tools'] = [self._tool_convertor(tool) for tool in tools]
        kwargs['betas'] = [_anthropic_tool_use_beta]

        res = self.__get_sync_client().beta.messages.create(**kwargs)

        tiny_res = anthropic_result_to_tiny_result(res)
        set_llm_telemetry_attributes(
            self.config, llm_input.messages, result=tiny_res.to_string(), tools=tools
        )
        return tiny_res

    @tiny_trace('agenerate_with_tools')
    async def agenerate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        kwargs = self.__create_client_kwargs(llm_input)
        kwargs['tools'] = [self._tool_convertor(tool) for tool in tools]
        kwargs['betas'] = [_anthropic_tool_use_beta]

        res = await self.__get_async_client().beta.messages.create(**kwargs)

        tiny_res = anthropic_result_to_tiny_result(res)
        set_llm_telemetry_attributes(
            self.config, llm_input.messages, result=tiny_res.to_string(), tools=tools
        )
        return tiny_res

    @tiny_trace('stream_with_tools')
    async def stream_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> AsyncIterator[TinyLLMResultChunk]:
        kwargs = self.__create_client_kwargs(llm_input)
        kwargs['tools'] = [self._tool_convertor(tool) for tool in tools]
        kwargs['extra_headers'] = {'anthropic-beta': _anthropic_tool_streaming_beta}

        set_llm_telemetry_attributes(self.config, llm_input.messages)

        async with self.__get_async_client().messages.stream(**kwargs) as stream:

            async def tiny_chunks() -> AsyncIterator[TinyLLMResultChunk]:
                async for text in stream.text_stream:
                    yield anthropic_chunk_to_tiny_chunk(
                        text
                    )  # here is yielding only TextBlock chunks (strings)

                final_message = await stream.get_final_message()
                tiny_final_message = anthropic_result_to_tiny_result(final_message)

                for generation in tiny_final_message.tiny_iter():
                    if isinstance(generation, TinyToolCall):
                        yield TinyLLMResultChunk(
                            type='tool_call', full_tool_call=generation
                        )

            accumulated_chunks: list[TinyLLMResultChunk] = []
            try:
                async for acc_chunk in tiny_chunks():
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

        kwargs = self.__create_client_kwargs(TinyLLMInput(messages=list(messages)))
        kwargs.pop('max_tokens')

        number_of_tokens = (
            self.__get_sync_client().messages.count_tokens(**kwargs).input_tokens
        )

        set_tiny_attribute('number_of_tokens', number_of_tokens)
        return number_of_tokens

    def __str__(self) -> str:
        buf = StringIO()

        buf.write('Claude LLM Summary:\n')
        buf.write(textwrap.indent(f'Model: {self.model}\n', '\t'))
        buf.write(textwrap.indent(f'Base URL: {self.base_url}\n', '\t'))
        buf.write(textwrap.indent(f'Timeout: {self.timeout}\n', '\t'))

        return buf.getvalue()
