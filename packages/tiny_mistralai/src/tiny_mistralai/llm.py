from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache
from io import StringIO
import json
import os
import textwrap
import typing
from typing import Iterable
from typing import Literal
from typing import override

from mistralai import Function
from mistralai import Mistral
from mistralai import Tool
from pydantic import Field
from pydantic import SecretStr
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase

from tiny_mistralai.utils import mistralai_chunk_to_tiny_chunks
from tiny_mistralai.utils import mistralai_family_to_tokenizer
from tiny_mistralai.utils import mistralai_result_to_tiny_result
from tiny_mistralai.utils import tiny_prompt_to_mistralai_params
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
    from tinygent.core.datamodels.llm import LLMStructuredT
    from tinygent.core.datamodels.tool import AbstractTool
    from tinygent.core.types.io.llm_io_input import TinyLLMInput
    from tinygent.core.types.io.llm_io_result import TinyLLMResult


class MistralAILLMConfig(AbstractLLMConfig['MistralAILLM']):
    type: Literal['mistralai'] = Field(default='mistralai', frozen=True)

    model: str = Field(default='mistral-medium-latest')

    api_key: SecretStr | None = Field(
        default_factory=lambda: (
            SecretStr(os.environ['MISTRALAI_API_KEY'])
            if 'MISTRALAI_API_KEY' in os.environ
            else None
        ),
    )

    safe_prompt: bool = Field(default=True)

    temperature: float = Field(default=0.6)

    timeout: float = Field(default=60.0)

    def build(self) -> MistralAILLM:
        return MistralAILLM(
            model=self.model,
            api_key=self.api_key.get_secret_value() if self.api_key else None,
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout=self.timeout,
        )


class MistralAILLM(AbstractLLM[MistralAILLMConfig]):
    def __init__(
        self,
        model: str = 'mistral-medium-latest',
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

        self._client: Mistral | None = None

        self.model = model
        self.api_key = api_key
        self.safe_prompt = safe_prompt
        self.temperature = temperature
        self.timeout = timeout

    @property
    def config(self) -> MistralAILLMConfig:
        return MistralAILLMConfig(
            model=self.model,
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout=self.timeout,
            api_key=SecretStr(self.api_key),
        )

    @property
    def supports_tool_calls(self) -> bool:
        return True  # INFO: Not all models may support tool calls, but mistralai api error if not.

    def __get_client(self) -> Mistral:
        if self._client:
            return self._client

        self._client = Mistral(api_key=self.api_key, timeout_ms=int(self.timeout) * 1000)
        return self._client

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

                prop = {'type': map_type(field_type)}
                if field.description:
                    prop['description'] = field.description

                properties[name] = prop

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

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_encoding(model: str) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(
            mistralai_family_to_tokenizer(model), use_fast=True
        )

    @staticmethod
    @lru_cache(maxsize=100_000)
    def _count_tokens(model: str, text: str) -> int:
        return len(MistralAILLM._get_encoding(model).encode(text))

    @tiny_trace('generate_text')
    def generate_text(
        self,
        llm_input: TinyLLMInput,
    ) -> TinyLLMResult:
        messages = tiny_prompt_to_mistralai_params(llm_input)

        res = self.__get_client().chat.complete(
            model=self.model,
            messages=messages,
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout_ms=int(self.timeout * 1000),
        )

        tiny_res = mistralai_result_to_tiny_result(res)
        set_llm_telemetry_attributes(
            self.config, llm_input.messages, result=tiny_res.to_string()
        )
        return tiny_res

    @tiny_trace('agenerate_text')
    async def agenerate_text(
        self,
        llm_input: TinyLLMInput,
    ) -> TinyLLMResult:
        messages = tiny_prompt_to_mistralai_params(llm_input)

        res = await self.__get_client().chat.complete_async(
            model=self.model,
            messages=messages,
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout_ms=int(self.timeout * 1000),
        )

        tiny_res = mistralai_result_to_tiny_result(res)
        set_llm_telemetry_attributes(
            self.config, llm_input.messages, result=tiny_res.to_string()
        )
        return tiny_res

    @tiny_trace('stream_text')
    async def stream_text(
        self, llm_input: TinyLLMInput
    ) -> AsyncIterator[TinyLLMResultChunk]:
        messages = tiny_prompt_to_mistralai_params(llm_input)
        set_llm_telemetry_attributes(self.config, llm_input.messages)

        res = await self.__get_client().chat.stream_async(
            model=self.model,
            messages=messages,
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout_ms=int(self.timeout * 1000),
        )

        async def raw_chunks() -> AsyncIterator[TinyLLMResultChunk]:
            async for chunk in res:
                for tiny_chunk in mistralai_chunk_to_tiny_chunks(chunk.data):
                    yield tiny_chunk

        accumulated_chunks: list[TinyLLMResultChunk] = []
        try:
            async for acc_chunk in accumulate_llm_chunks(raw_chunks()):
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
        messages = tiny_prompt_to_mistralai_params(llm_input)

        res = self.__get_client().chat.parse(
            model=self.model,
            messages=messages,
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout_ms=int(self.timeout * 1000),
            response_format=output_schema,
        )

        if not res.choices or not (message := res.choices[0].message):
            raise ValueError('No message in MistralAI response.')

        parsed = output_schema.model_validate(json.loads(str(message.content) or '{}'))
        set_llm_telemetry_attributes(
            self.config,
            llm_input.messages,
            result=str(parsed),
            output_schema=output_schema,
        )
        return parsed

    @tiny_trace('agenerate_structured')
    async def agenerate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT:
        messages = tiny_prompt_to_mistralai_params(llm_input)

        res = await self.__get_client().chat.parse_async(
            model=self.model,
            messages=messages,
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout_ms=int(self.timeout * 1000),
            response_format=output_schema,
        )

        if not res.choices or not (message := res.choices[0].message):
            raise ValueError('No message in MistralAI response.')

        parsed = output_schema.model_validate(json.loads(str(message.content) or '{}'))
        set_llm_telemetry_attributes(
            self.config,
            llm_input.messages,
            result=str(parsed),
            output_schema=output_schema,
        )
        return parsed

    @tiny_trace('generate_with_tools')
    def generate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        functions = [self._tool_convertor(tool) for tool in tools]
        messages = tiny_prompt_to_mistralai_params(llm_input)

        res = self.__get_client().chat.complete(
            model=self.model,
            messages=messages,
            tools=functions,
            tool_choice='auto',
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout_ms=int(self.timeout * 1000),
        )

        tiny_res = mistralai_result_to_tiny_result(res)
        set_llm_telemetry_attributes(
            self.config, llm_input.messages, result=tiny_res.to_string(), tools=tools
        )
        return tiny_res

    @tiny_trace('agenerate_with_tools')
    async def agenerate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        functions = [self._tool_convertor(tool) for tool in tools]
        messages = tiny_prompt_to_mistralai_params(llm_input)

        res = await self.__get_client().chat.complete_async(
            model=self.model,
            messages=messages,
            tools=functions,
            tool_choice='auto',
            safe_prompt=self.safe_prompt,
            temperature=self.temperature,
            timeout_ms=int(self.timeout * 1000),
        )

        tiny_res = mistralai_result_to_tiny_result(res)
        set_llm_telemetry_attributes(
            self.config, llm_input.messages, result=tiny_res.to_string(), tools=tools
        )
        return tiny_res

    @tiny_trace('stream_with_tools')
    async def stream_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> AsyncIterator[TinyLLMResultChunk]:
        functions = [self._tool_convertor(tool) for tool in tools]
        messages = tiny_prompt_to_mistralai_params(llm_input)
        set_llm_telemetry_attributes(self.config, llm_input.messages, tools=tools)

        res = await self.__get_client().chat.stream_async(
            model=self.model,
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

        accumulated_chunks: list[TinyLLMResultChunk] = []
        try:
            async for acc_chunk in accumulate_llm_chunks(raw_chunks()):
                accumulated_chunks.append(acc_chunk)
                yield acc_chunk
        finally:
            set_tiny_attribute(
                'result',
                group_chunks_for_telemetry(accumulated_chunks),
            )

    def count_tokens_in_messages(self, messages: Iterable[AllTinyMessages]) -> int:
        set_llm_telemetry_attributes(self.config, messages)

        number_of_tokens = sum(
            [MistralAILLM._count_tokens(self.model, m.tiny_str) for m in messages]
        )

        set_tiny_attribute('number_of_tokens', number_of_tokens)
        return number_of_tokens

    def __str__(self) -> str:
        buf = StringIO()

        buf.write('MistralAI LLM Summary:\n')
        buf.write(textwrap.indent(f'Model: {self.model}\n', '\t'))
        buf.write(textwrap.indent(f'Safe Prompt: {self.safe_prompt}\n', '\t'))
        buf.write(textwrap.indent(f'Temperature: {self.temperature}\n', '\t'))
        buf.write(textwrap.indent(f'Timeout: {self.timeout}\n', '\t'))

        return buf.getvalue()
