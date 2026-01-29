from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from io import StringIO
import logging
import textwrap
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Generic
from typing import Sequence
from typing import TypeVar

from pydantic import Field

from tinygent.agents.middleware.tool_limiter import ToolCallBlockedException
from tinygent.core.datamodels.agent import AbstractAgent
from tinygent.core.datamodels.agent import AbstractAgentConfig
from tinygent.core.datamodels.llm import AbstractLLM
from tinygent.core.datamodels.llm import AbstractLLMConfig
from tinygent.core.datamodels.memory import AbstractMemory
from tinygent.core.datamodels.memory import AbstractMemoryConfig
from tinygent.core.datamodels.messages import TinyToolCall
from tinygent.core.datamodels.messages import TinyToolResult
from tinygent.core.datamodels.middleware import AbstractMiddleware
from tinygent.core.datamodels.middleware import AbstractMiddlewareConfig
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.datamodels.tool import AbstractToolConfig
from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.otel import set_tiny_attribute
from tinygent.core.telemetry.otel import set_tiny_attributes
from tinygent.core.types.io.llm_io_chunks import TinyLLMResultChunk
from tinygent.core.types.io.llm_io_input import TinyLLMInput
from tinygent.memory.buffer_chat_memory import BufferChatMemoryConfig

T = TypeVar('T', bound='AbstractAgent')


logger = logging.getLogger(__name__)


class TinyBaseAgentConfig(AbstractAgentConfig[T], Generic[T]):
    """Configuration for BaseAgent."""

    type: Any = Field(default='base')

    middleware: Sequence[AbstractMiddlewareConfig | AbstractMiddleware] = Field(
        default_factory=list
    )

    llm: AbstractLLMConfig | AbstractLLM = Field(...)
    tools: Sequence[AbstractToolConfig | AbstractTool] = Field(default_factory=list)
    memory: AbstractMemoryConfig | AbstractMemory = Field(
        default_factory=BufferChatMemoryConfig
    )

    def build(self) -> T:
        """Build the BaseAgent instance from the configuration."""
        raise NotImplementedError('Subclasses must implement this method.')

    def build_llm_instance(self) -> AbstractLLM:
        """Build LLM instance from config or return existing instance."""
        if isinstance(self.llm, AbstractLLM):
            return self.llm
        from tinygent.core.factory.llm import build_llm

        return build_llm(self.llm)

    def build_tools_list(self) -> list[AbstractTool]:
        """Build list of tool instances from configs or return existing instances."""
        from tinygent.core.factory.tool import build_tool

        return [
            tool if isinstance(tool, AbstractTool) else build_tool(tool)
            for tool in self.tools
        ]

    def build_memory_instance(self) -> AbstractMemory:
        """Build memory instance from config or return existing instance."""
        if isinstance(self.memory, AbstractMemory):
            return self.memory
        from tinygent.core.factory.memory import build_memory

        return build_memory(self.memory)

    def build_middleware_list(self) -> list[AbstractMiddleware]:
        """Build list of middleware instances from configs or return existing instances."""
        from tinygent.core.factory.middleware import build_middleware

        return [
            m if isinstance(m, AbstractMiddleware) else build_middleware(m)
            for m in self.middleware
        ]


class TinyBaseAgent(AbstractAgent, AbstractMiddleware):
    def __init__(
        self,
        llm: AbstractLLM,
        memory: AbstractMemory,
        tools: Sequence[AbstractTool] = (),
        middleware: Sequence[AbstractMiddleware] = [],
    ) -> None:
        self.llm = llm
        self.middleware = middleware

        self._memory = memory
        self._tools = tools
        self._final_answer: str | None = None

    def reset(self) -> None:
        logger.debug('[BASE AGENT RESET]')

        self.memory.clear()
        self._final_answer = None

    @property
    def memory(self) -> AbstractMemory:
        return self._memory

    @property
    def tools(self) -> Sequence[AbstractTool]:
        return self._tools

    def get_tool(self, name: str) -> AbstractTool | None:
        logger.debug('Looking for tool: %s', name)
        tool = next((tool for tool in self.tools if tool.info.name == name), None)
        if tool:
            logger.debug('Tool %s founded => %s', name, tool)
        else:
            logger.warning('Tool %s not found.', name)
        return tool

    def get_tool_from_list(
        self, tools: list[AbstractTool], name: str
    ) -> AbstractTool | None:
        logger.debug(
            'Looking for tool: %s amongst %s', name, [t.info.name for t in tools]
        )
        tool = next((tool for tool in tools if tool.info.name == name), None)
        if tool:
            logger.debug('Tool %s founded => %s', name, tool)
        else:
            logger.warning('Tool %s not found.', name)
        return tool

    @tiny_trace('run_llm')
    async def run_llm(
        self, run_id: str, fn: Callable, llm_input: TinyLLMInput, **kwargs
    ) -> Any:
        kwargs_dict = dict(kwargs)
        await self.before_llm_call(
            run_id=run_id, llm_input=llm_input, kwargs=kwargs_dict
        )
        try:
            result = fn(llm_input=llm_input, **kwargs_dict)
            await self.after_llm_call(
                run_id=run_id, llm_input=llm_input, result=result, kwargs=kwargs_dict
            )
            return result
        except Exception as e:
            logger.warning('Error during llm call: %s', e)
            await self.on_error(run_id=run_id, e=e, kwargs=kwargs_dict)
            raise

    @tiny_trace('run_llm_stream')
    async def run_llm_stream(
        self,
        run_id: str,
        fn: Callable[
            ...,
            AsyncIterator[TinyLLMResultChunk]
            | Awaitable[AsyncIterator[TinyLLMResultChunk]],
        ],
        llm_input: TinyLLMInput,
        **kwargs: Any,
    ) -> AsyncGenerator[TinyLLMResultChunk, None]:
        kwargs_dict = dict(kwargs)
        await self.before_llm_call(
            run_id=run_id, llm_input=llm_input, kwargs=kwargs_dict
        )
        try:
            result = fn(llm_input=llm_input, **kwargs_dict)

            # if coroutine, await it to get async generator
            if isinstance(result, Awaitable):
                result = await result

            all_chunks = []
            async for chunk in result:
                all_chunks.append(chunk)
                yield chunk

            await self.after_llm_call(
                run_id=run_id, llm_input=llm_input, result=None, kwargs=kwargs_dict
            )
        except Exception as e:
            logger.warning('Error during llm stream call: %s', e)
            await self.on_error(run_id=run_id, e=e, kwargs=kwargs_dict)
            raise

    @tiny_trace('run_tool')
    async def run_tool(
        self, run_id: str, tool: AbstractTool, call: TinyToolCall, **kwargs: Any
    ) -> TinyToolResult:
        set_tiny_attributes(
            {
                'tool.name': tool.info.name,
                'tool.arguments': str(call.arguments),
            }
        )
        logger.debug('Running tool %s(%s)', tool.info.name, call.arguments)

        kwargs_dict = dict(kwargs)
        try:
            await self.before_tool_call(
                run_id=run_id, tool=tool, args=call.arguments, kwargs=kwargs_dict
            )

            result = tool(**call.arguments)
            call.metadata['executed'] = True
            call.result = result
            await self.after_tool_call(
                run_id=run_id,
                tool=tool,
                args=call.arguments,
                result=result,
                kwargs=kwargs_dict,
            )

            tool_result = TinyToolResult(
                call_id=call.call_id or 'unknown',
                content=str(result),
            )

            set_tiny_attribute('tool.result', str(result))
            logger.debug(
                'Tool %s(%s) => %s', tool.info.name, call.arguments, str(result)
            )

            tool_result.raw = tool
            return tool_result
        except ToolCallBlockedException as e:
            logger.warning(
                'Tool call blocked by middleware: %s(%s) - %s',
                tool.info.name,
                call.arguments,
                str(e),
            )
            await self.on_error(run_id=run_id, e=e, kwargs=kwargs_dict)

            # Return error result to maintain tool call/result consistency
            error_result = TinyToolResult(
                call_id=call.call_id or 'unknown',
                content=f'Tool call blocked: {str(e)}',
            )
            call.metadata['executed'] = False
            call.metadata['blocked'] = True
            call.metadata['error'] = e

            set_tiny_attribute('tool.blocked', str(e))
            return error_result
        except Exception as e:
            logger.warning(
                'Error during tool call %s(%s)', tool.info.name, call.arguments
            )
            await self.on_error(run_id=run_id, e=e, kwargs=kwargs_dict)
            raise

    def __str__(self) -> str:
        buf = StringIO()

        buf.write('\n')
        buf.write('Agent Summary:\n')
        buf.write(f'{textwrap.indent(str(self.llm), "\t")}\n')

        buf.write('\tMemory:\n')
        buf.write(f'{textwrap.indent(str(self.memory), "\t\t")}\n')

        buf.write(f'\tTools ({len(self.tools)}):\n')
        if len(self.tools) > 0:
            for tool in self.tools:
                buf.write(f'{textwrap.indent(str(tool), "\t\t")}\n')
        else:
            buf.write('\t\tNo tools configured.\n')

        buf.write(f'\tMiddlewares ({len(self.middleware)}):\n')
        if len(self.middleware) > 0:
            for middleware in self.middleware:
                buf.write(f'{textwrap.indent(str(middleware), "\t\t")}\n')
        else:
            buf.write('\t\tNo middlewares configured.\n')

        return buf.getvalue()
