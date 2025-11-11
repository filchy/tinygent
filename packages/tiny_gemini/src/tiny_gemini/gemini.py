from __future__ import annotations

from collections.abc import AsyncIterator
import os
import typing
from typing import Literal

from google import genai
from google.genai.types import FunctionDeclaration
from google.genai.types import Schema
from google.genai.types import Tool
from google.genai.types import Type

from tiny_gemini.utils import gemini_chunk_to_tiny_chunks
from tiny_gemini.utils import gemini_response_to_tiny_result
from tiny_gemini.utils import tiny_attributes_to_gemini_config
from tiny_gemini.utils import tiny_prompt_to_gemini_params
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm import AbstractLLMConfig
from tinygent.datamodels.llm_io_chunks import TinyLLMResultChunk

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm import LLMStructuredT
    from tinygent.datamodels.llm_io_input import TinyLLMInput
    from tinygent.datamodels.llm_io_result import TinyLLMResult
    from tinygent.datamodels.tool import AbstractTool


class GeminiConfig(AbstractLLMConfig['GeminiLLM']):
    type: Literal['gemini'] = 'gemini'

    model: str = 'gemini-2.5-flash'

    temperature: float = 0.6

    api_key: str | None = os.getenv('GEMINI_API_KEY', None)

    def build(self) -> GeminiLLM:
        return GeminiLLM(
            model_name=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
        )


class GeminiLLM(AbstractLLM[GeminiConfig]):
    def __init__(
        self,
        model_name: str = 'gemini-2.5-flash',
        temperature: float = 0.6,
        api_key: str | None = None,
    ) -> None:
        if not api_key and not (api_key := os.getenv('GEMINI_API_KEY')):
            raise ValueError(
                'Gemini API key must be provided either via config',
                " or 'GEMINI_API_KEY' env variable.",
            )

        self._sync_client = genai.Client(api_key=api_key)

        self._async_client = self._sync_client.aio

        self.model_name = model_name
        self.temperature = temperature

    @property
    def supports_tool_calls(self) -> bool:
        return True

    def _tool_convertor(self, tool: AbstractTool) -> Tool:
        info = tool.info
        schema = info.input_schema

        def map_type(py_type: type) -> Type:
            mapping = {
                str: Type.STRING,
                int: Type.INTEGER,
                float: Type.NUMBER,
                bool: Type.BOOLEAN,
                list: Type.ARRAY,
                dict: Type.OBJECT,
            }
            return mapping.get(py_type, Type.STRING)  # default fallback

        properties = {}

        if schema:
            for name, field in schema.model_fields.items():
                field_type = (
                    field.annotation
                    if isinstance(field.annotation, type)
                    else type(field.annotation)
                )
                properties[name] = Schema(
                    type=map_type(field_type),
                    description=field.description,
                )

        func_declaration = FunctionDeclaration(
            name=info.name,
            description=info.description,
            parameters=Schema(
                type=Type.OBJECT,
                properties=properties,
                required=info.required_fields,
            ),
        )

        return Tool(
            function_declarations=[func_declaration]
        )

    def generate_text(self, llm_input: TinyLLMInput) -> TinyLLMResult:
        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(llm_input, self.temperature)

        chat = self._sync_client.chats.create(
            model=self.model_name,
            config=config,
            history=params['history'],
        )
        res = chat.send_message(params['message'])

        return gemini_response_to_tiny_result(res)

    async def agenerate_text(
        self,
        llm_input: TinyLLMInput,
    ) -> TinyLLMResult:
        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(llm_input, self.temperature)

        chat = self._async_client.chats.create(
            model=self.model_name,
            config=config,
            history=params['history'],
        )
        res = await chat.send_message(params['message'])

        return gemini_response_to_tiny_result(res)

    async def stream_text(
        self, llm_input: TinyLLMInput
    ) -> AsyncIterator[TinyLLMResultChunk]:
        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(llm_input, self.temperature)

        chat = self._async_client.chats.create(
            model=self.model_name,
            config=config,
            history=params['history'],
        )
        res = await chat.send_message_stream(params['message'])
        async for chunk in res:
            for tiny_chunk in gemini_chunk_to_tiny_chunks(chunk):
                yield tiny_chunk

    def generate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT:
        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(
            llm_input,
            self.temperature,
            structured_output=output_schema,
        )

        chat = self._sync_client.chats.create(
            model=self.model_name,
            config=config,
            history=params['history'],
        )
        res = chat.send_message(params['message'])
        tiny_result = gemini_response_to_tiny_result(res)
        for message in tiny_result.tiny_iter():
            if (content := getattr(message, 'content', None)):
                try:
                    return output_schema.model_validate_json(content)
                except Exception:
                    pass

        raise ValueError('No valid structured output found in Gemini response.')

    async def agenerate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT:
        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(
            llm_input,
            self.temperature,
            structured_output=output_schema,
        )

        chat = self._async_client.chats.create(
            model=self.model_name,
            config=config,
            history=params['history'],
        )
        res = await chat.send_message(params['message'])
        tiny_result = gemini_response_to_tiny_result(res)
        for message in tiny_result.tiny_iter():
            if (content := getattr(message, 'content', None)):
                try:
                    return output_schema.model_validate_json(content)
                except Exception:
                    pass

        raise ValueError('No valid structured output found in Gemini response.')

    def generate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        gemini_tools = [self._tool_convertor(tool) for tool in tools]

        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(
            llm_input,
            self.temperature,
            tools=gemini_tools
        )

        chat = self._sync_client.chats.create(
            model=self.model_name,
            config=config,
            history=params['history'],
        )
        res = chat.send_message(params['message'])

        return gemini_response_to_tiny_result(res)

    async def agenerate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        gemini_tools = [self._tool_convertor(tool) for tool in tools]

        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(
            llm_input,
            self.temperature,
            tools=gemini_tools
        )

        chat = self._async_client.chats.create(
            model=self.model_name,
            config=config,
            history=params['history'],
        )
        res = await chat.send_message(params['message'])

        return gemini_response_to_tiny_result(res)

    async def stream_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> AsyncIterator[TinyLLMResultChunk]:
        gemini_tools = [self._tool_convertor(tool) for tool in tools]

        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(
            llm_input,
            self.temperature,
            tools=gemini_tools
        )

        chat = self._async_client.chats.create(
            model=self.model_name,
            config=config,
            history=params['history'],
        )
        res = await chat.send_message_stream(params['message'])
        async for chunk in res:
            for tiny_chunk in gemini_chunk_to_tiny_chunks(chunk):
                yield tiny_chunk
