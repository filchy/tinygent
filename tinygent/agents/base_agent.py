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

from tinygent.agents.middleware.base import AgentMiddleware
from tinygent.agents.middleware.agent import MiddlewareAgent
from tinygent.datamodels.agent import AbstractAgent
from tinygent.datamodels.agent import AbstractAgentConfig
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm import AbstractLLMConfig
from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.memory import AbstractMemoryConfig
from tinygent.datamodels.messages import TinyToolCall
from tinygent.datamodels.messages import TinyToolResult
from tinygent.datamodels.tool import AbstractTool
from tinygent.datamodels.tool import AbstractToolConfig
from tinygent.memory.buffer_chat_memory import BufferChatMemoryConfig
from tinygent.telemetry.decorators import tiny_trace
from tinygent.telemetry.otel import set_tiny_attribute
from tinygent.telemetry.otel import set_tiny_attributes
from tinygent.types.io.llm_io_chunks import TinyLLMResultChunk
from tinygent.types.io.llm_io_input import TinyLLMInput

T = TypeVar('T', bound='AbstractAgent')


logger = logging.getLogger(__name__)


class TinyBaseAgentConfig(AbstractAgentConfig[T], Generic[T]):
    """Configuration for BaseAgent."""

    type: Any = 'base'

    middleware: Sequence[AgentMiddleware] = Field(default_factory=list)

    llm: AbstractLLMConfig | AbstractLLM
    tools: Sequence[AbstractToolConfig | AbstractTool] = Field(default_factory=list)
    memory: AbstractMemoryConfig | AbstractMemory = Field(
        default_factory=BufferChatMemoryConfig
    )

    def build(self) -> T:
        """Build the BaseAgent instance from the configuration."""
        raise NotImplementedError('Subclasses must implement this method.')


class TinyBaseAgent(AbstractAgent, MiddlewareAgent):
    def __init__(
        self,
        llm: AbstractLLM,
        memory: AbstractMemory,
        tools: Sequence[AbstractTool] = (),
        middleware: Sequence[AgentMiddleware] = [],
    ) -> None:
        MiddlewareAgent.__init__(self, middleware)
        self.llm = llm
        self.memory = memory

        self._tools = tools
        self._final_answer: str | None = None

    def reset(self) -> None:
        logger.debug('[BASE AGENT RESET]')

        self.memory.clear()
        self._final_answer = None

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
    def run_llm(
        self, run_id: str, fn: Callable, llm_input: TinyLLMInput, **kwargs
    ) -> Any:
        self.before_llm_call(run_id=run_id, llm_input=llm_input)
        try:
            result = fn(llm_input=llm_input, **kwargs)
            self.after_llm_call(run_id=run_id, llm_input=llm_input, result=result)
            return result
        except Exception as e:
            logger.warning('Error during llm call: %s', e)
            self.on_error(run_id=run_id, e=e)
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
        self.before_llm_call(run_id=run_id, llm_input=llm_input)
        try:
            result = fn(llm_input=llm_input, **kwargs)

            # if coroutine, await it to get async generator
            if isinstance(result, Awaitable):
                result = await result

            all_chunks = []
            async for chunk in result:
                all_chunks.append(chunk)
                yield chunk

            self.after_llm_call(run_id=run_id, llm_input=llm_input, result=None)
        except Exception as e:
            logger.warning('Error during llm stream call: %s', e)
            self.on_error(run_id=run_id, e=e)
            raise

    @tiny_trace('run_tool')
    def run_tool(
        self, run_id: str, tool: AbstractTool, call: TinyToolCall
    ) -> TinyToolResult:
        set_tiny_attributes(
            {
                'tool.name': tool.info.name,
                'tool.arguments': str(call.arguments),
            }
        )
        logger.debug('Running tool %s(%s)', tool.info.name, call.arguments)

        self.before_tool_call(run_id=run_id, tool=tool, args=call.arguments)
        try:
            result = tool(**call.arguments)
            call.metadata['executed'] = True
            call.result = result
            self.after_tool_call(
                run_id=run_id, tool=tool, args=call.arguments, result=result
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
        except Exception as e:
            logger.warning(
                'Error during tool call %s(%s)', tool.info.name, call.arguments
            )
            self.on_error(run_id=run_id, e=e)
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

        return buf.getvalue()
