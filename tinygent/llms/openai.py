from __future__ import annotations

from collections import defaultdict
from collections.abc import AsyncGenerator
from io import StringIO
import json
import os
import textwrap
import typing
from typing import Any
from typing import Literal
from typing import override

from openai import AsyncOpenAI
from openai import OpenAI
from openai.lib.streaming.chat import ChunkEvent
from openai.types.chat import ChatCompletionFunctionToolParam

from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm import AbstractLLMConfig
from tinygent.datamodels.llm_io_chunks import TinyLLMResultChunk
from tinygent.datamodels.messages import TinyToolCall
from tinygent.llms.utils import openai_chunk_to_tiny_chunk
from tinygent.llms.utils import openai_result_to_tiny_result
from tinygent.llms.utils import tiny_prompt_to_openai_params

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm import LLMStructuredT
    from tinygent.datamodels.llm_io_input import TinyLLMInput
    from tinygent.datamodels.llm_io_result import TinyLLMResult
    from tinygent.datamodels.tool import AbstractTool


class OpenAIConfig(AbstractLLMConfig['OpenAILLM']):
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


class OpenAILLM(AbstractLLM[OpenAIConfig]):
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

        self._sync_client = OpenAI(api_key=api_key, base_url=base_url)

        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout

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

    def generate_text(
        self,
        llm_input: TinyLLMInput,
    ) -> TinyLLMResult:
        messages = tiny_prompt_to_openai_params(llm_input)

        res = self._sync_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            timeout=self.timeout,
        )

        return openai_result_to_tiny_result(res)

    async def agenerate_text(self, llm_input: TinyLLMInput) -> TinyLLMResult:
        messages = tiny_prompt_to_openai_params(llm_input)

        res = await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            timeout=self.timeout,
        )

        return openai_result_to_tiny_result(res)

    async def stream_text(  # type: ignore[override]
        self, llm_input: TinyLLMInput
    ) -> AsyncGenerator[TinyLLMResultChunk, None]:
        messages = tiny_prompt_to_openai_params(llm_input)

        async with self._async_client.chat.completions.stream(
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

        res = self._sync_client.chat.completions.parse(
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

        res = await self._async_client.chat.completions.parse(
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

        res = self._sync_client.chat.completions.create(
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

        res = await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=functions,
            tool_choice='auto',
            temperature=self.temperature,
            timeout=self.timeout,
        )

        return openai_result_to_tiny_result(res)

    async def stream_with_tools(  # type: ignore[override]
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> AsyncGenerator[TinyLLMResultChunk, None]:
        functions = [self._tool_convertor(tool) for tool in tools]
        messages = tiny_prompt_to_openai_params(llm_input)

        pending_tool_calls: dict[int, dict[str, Any]] = defaultdict(
            lambda: {'id': '', 'name': '', 'args': []}
        )

        async with self._async_client.chat.completions.stream(
            model=self.model_name,
            messages=messages,
            tools=functions,
            tool_choice='auto',
            temperature=self.temperature,
            timeout=self.timeout,
        ) as stream:
            async for event in stream:
                if not isinstance(event, ChunkEvent):
                    continue

                tiny_chunk = openai_chunk_to_tiny_chunk(event.chunk)

                # message chunk
                if tiny_chunk.is_message and tiny_chunk.message:
                    yield tiny_chunk

                # tool call chunk
                elif tiny_chunk.is_tool_call and tiny_chunk.tool_call:
                    tc = tiny_chunk.tool_call
                    state = pending_tool_calls[tc.index]

                    if tc.call_id and not state['id']:
                        state['id'] = tc.call_id
                    if tc.tool_name and not state['name']:
                        state['name'] = tc.tool_name

                    if tc.arguments:
                        state['args'].append(tc.arguments)

                        try:
                            args_str = ''.join(state['args'])
                            tool_args = json.loads(args_str)
                        except json.JSONDecodeError:
                            continue  # wait for more chunks

                        yield TinyLLMResultChunk(
                            type='tool_call',
                            full_tool_call=TinyToolCall(
                                tool_name=state['name'],
                                arguments=tool_args,
                                call_id=state['id'] or None,
                                metadata={'raw': state},
                            ),
                            metadata=tiny_chunk.metadata,
                        )

    def __str__(self) -> str:
        buf = StringIO()

        buf.write('OpenAI LLM Summary:\n')
        buf.write(textwrap.indent(f'Model: {self.model_name}\n', '\t'))
        buf.write(textwrap.indent(f'Base URL: {self.base_url}\n', '\t'))
        buf.write(textwrap.indent(f'Temperature: {self.temperature}\n', '\t'))
        buf.write(textwrap.indent(f'Timeout: {self.timeout}\n', '\t'))
        return buf.getvalue()
