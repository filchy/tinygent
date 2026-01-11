from __future__ import annotations

from collections.abc import AsyncIterator
import os
import typing
from typing import Literal

from google.genai.client import AsyncClient
from google.genai.client import Client
from google.genai.types import FunctionDeclarationDict
from google.genai.types import SchemaDict
from google.genai.types import ToolDict
from google.genai.types import Type
from pydantic import Field
from pydantic import SecretStr

from tiny_gemini.utils import gemini_chunk_to_tiny_chunks
from tiny_gemini.utils import gemini_response_to_tiny_result
from tiny_gemini.utils import tiny_attributes_to_gemini_config
from tiny_gemini.utils import tiny_prompt_to_gemini_params
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm import AbstractLLMConfig
from tinygent.llms.utils import accumulate_llm_chunks
from tinygent.llms.utils import group_chunks_for_telemetry
from tinygent.telemetry.decorators import tiny_trace
from tinygent.telemetry.otel import set_tiny_attribute
from tinygent.telemetry.utils import set_llm_telemetry_attributes
from tinygent.types.io.llm_io_chunks import TinyLLMResultChunk

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm import LLMStructuredT
    from tinygent.datamodels.tool import AbstractTool
    from tinygent.types.io.llm_io_input import TinyLLMInput
    from tinygent.types.io.llm_io_result import TinyLLMResult


class GeminiLLMConfig(AbstractLLMConfig['GeminiLLM']):
    type: Literal['gemini'] = Field(default='gemini', frozen=True)

    model: str = Field(default='gemini-2.5-flash')

    temperature: float = Field(default=0.6)

    api_key: SecretStr | None = Field(
        default_factory=lambda: (
            SecretStr(os.environ['GEMINI_API_KEY'])
            if 'GEMINI_API_KEY' in os.environ
            else None
        ),
    )

    def build(self) -> GeminiLLM:
        return GeminiLLM(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key.get_secret_value() if self.api_key else None,
        )


class GeminiLLM(AbstractLLM[GeminiLLMConfig]):
    def __init__(
        self,
        model: str = 'gemini-2.5-flash',
        temperature: float = 0.6,
        api_key: str | None = None,
    ) -> None:
        if not api_key and not (api_key := os.getenv('GEMINI_API_KEY')):
            raise ValueError(
                'Gemini API key must be provided either via config',
                " or 'GEMINI_API_KEY' env variable.",
            )

        self._sync_client: Client | None = Client(api_key=api_key)

        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    @property
    def config(self) -> GeminiLLMConfig:
        return GeminiLLMConfig(
            model=self.model,
            temperature=self.temperature,
            api_key=SecretStr(self.api_key),
        )

    @property
    def supports_tool_calls(self) -> bool:
        return True

    def __get_sync_client(self) -> Client:
        if self._sync_client:
            return self._sync_client

        self._sync_client = Client(api_key=self.api_key)
        return self._sync_client

    def __get_async_client(self) -> AsyncClient:
        cli = self.__get_sync_client()
        return cli.aio

    def _tool_convertor(self, tool: AbstractTool) -> ToolDict:
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
                properties[name] = SchemaDict(
                    type=map_type(field_type),
                    description=field.description,
                )

        func_declaration = FunctionDeclarationDict(
            name=info.name,
            description=info.description,
            parameters=SchemaDict(
                type=Type.OBJECT,
                properties=properties,
                required=info.required_fields,
            ),
        )

        return ToolDict(function_declarations=[func_declaration])

    @tiny_trace('generate_text')
    def generate_text(self, llm_input: TinyLLMInput) -> TinyLLMResult:
        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(llm_input, self.temperature)

        chat = self.__get_sync_client().chats.create(
            model=self.model,
            config=config,
            history=params['history'],
        )
        res = chat.send_message(params['message'])  # type: ignore

        tiny_res = gemini_response_to_tiny_result(res)
        set_llm_telemetry_attributes(self.config, llm_input, result=tiny_res.to_string())
        return tiny_res

    @tiny_trace('agenerate_text')
    async def agenerate_text(
        self,
        llm_input: TinyLLMInput,
    ) -> TinyLLMResult:
        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(llm_input, self.temperature)

        chat = self.__get_async_client().chats.create(
            model=self.model,
            config=config,
            history=params['history'],
        )
        res = await chat.send_message(params['message'])  # type: ignore

        tiny_res = gemini_response_to_tiny_result(res)
        set_llm_telemetry_attributes(self.config, llm_input, result=tiny_res.to_string())
        return tiny_res

    @tiny_trace('stream_text')
    async def stream_text(
        self, llm_input: TinyLLMInput
    ) -> AsyncIterator[TinyLLMResultChunk]:
        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(llm_input, self.temperature)
        set_llm_telemetry_attributes(self.config, llm_input)

        chat = self.__get_async_client().chats.create(
            model=self.model,
            config=config,
            history=params['history'],
        )
        res = await chat.send_message_stream(params['message'])  # type: ignore

        async def raw_chunks() -> AsyncIterator[TinyLLMResultChunk]:
            async for chunk in res:
                for tiny_chunk in gemini_chunk_to_tiny_chunks(chunk):
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
        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(
            llm_input,
            self.temperature,
            structured_output=output_schema,
        )

        chat = self.__get_sync_client().chats.create(
            model=self.model,
            config=config,
            history=params['history'],
        )
        res = chat.send_message(params['message'])  # type: ignore
        tiny_result = gemini_response_to_tiny_result(res)
        for message in tiny_result.tiny_iter():
            if content := getattr(message, 'content', None):
                try:
                    parsed = output_schema.model_validate_json(content)
                    set_llm_telemetry_attributes(
                        self.config,
                        llm_input,
                        result=str(parsed),
                        output_schema=output_schema,
                    )
                    return parsed
                except Exception:
                    pass

        raise ValueError('No valid structured output found in Gemini response.')

    @tiny_trace('agenerate_structured')
    async def agenerate_structured(
        self, llm_input: TinyLLMInput, output_schema: type[LLMStructuredT]
    ) -> LLMStructuredT:
        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(
            llm_input,
            self.temperature,
            structured_output=output_schema,
        )

        chat = self.__get_async_client().chats.create(
            model=self.model,
            config=config,
            history=params['history'],
        )
        res = await chat.send_message(params['message'])  # type: ignore
        tiny_result = gemini_response_to_tiny_result(res)
        for message in tiny_result.tiny_iter():
            if content := getattr(message, 'content', None):
                try:
                    parsed = output_schema.model_validate_json(content)
                    set_llm_telemetry_attributes(
                        self.config,
                        llm_input,
                        result=str(parsed),
                        output_schema=output_schema,
                    )
                    return parsed
                except Exception:
                    pass

        raise ValueError('No valid structured output found in Gemini response.')

    @tiny_trace('generate_with_tools')
    def generate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        gemini_tools = [self._tool_convertor(tool) for tool in tools]

        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(
            llm_input,
            self.temperature,
            tools=gemini_tools,  # type: ignore
        )

        chat = self.__get_sync_client().chats.create(
            model=self.model,
            config=config,
            history=params['history'],
        )
        res = chat.send_message(params['message'])  # type: ignore

        tiny_res = gemini_response_to_tiny_result(res)
        set_llm_telemetry_attributes(
            self.config, llm_input, result=tiny_res.to_string(), tools=tools
        )
        return tiny_res

    @tiny_trace('agenerate_with_tools')
    async def agenerate_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> TinyLLMResult:
        gemini_tools = [self._tool_convertor(tool) for tool in tools]

        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(
            llm_input,
            self.temperature,
            tools=gemini_tools,  # type: ignore
        )

        chat = self.__get_async_client().chats.create(
            model=self.model,
            config=config,
            history=params['history'],
        )
        res = await chat.send_message(params['message'])  # type: ignore

        tiny_res = gemini_response_to_tiny_result(res)
        set_llm_telemetry_attributes(
            self.config, llm_input, result=tiny_res.to_string(), tools=tools
        )
        return tiny_res

    @tiny_trace('stream_with_tools')
    async def stream_with_tools(
        self, llm_input: TinyLLMInput, tools: list[AbstractTool]
    ) -> AsyncIterator[TinyLLMResultChunk]:
        gemini_tools = [self._tool_convertor(tool) for tool in tools]

        params = tiny_prompt_to_gemini_params(llm_input)
        config = tiny_attributes_to_gemini_config(
            llm_input,
            self.temperature,
            tools=gemini_tools,  # type: ignore
        )
        set_llm_telemetry_attributes(self.config, llm_input, tools=tools)

        chat = self.__get_async_client().chats.create(
            model=self.model,
            config=config,
            history=params['history'],
        )
        res = await chat.send_message_stream(params['message'])  # type: ignore

        async def raw_chunks() -> AsyncIterator[TinyLLMResultChunk]:
            async for chunk in res:
                for tiny_chunk in gemini_chunk_to_tiny_chunks(chunk):
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
