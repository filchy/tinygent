from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from io import StringIO
import textwrap
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Generic
from typing import Sequence
from typing import TypeVar

from pydantic import Field

from tinygent.datamodels.agent import AbstractAgent
from tinygent.datamodels.agent import AbstractAgentConfig
from tinygent.datamodels.agent_hooks import AgentHooks
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm import AbstractLLMConfig
from tinygent.datamodels.llm_io_chunks import TinyLLMResultChunk
from tinygent.datamodels.llm_io_input import TinyLLMInput
from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.memory import AbstractMemoryConfig
from tinygent.datamodels.messages import TinyToolCall
from tinygent.datamodels.messages import TinyToolResult
from tinygent.datamodels.tool import AbstractTool
from tinygent.datamodels.tool import AbstractToolConfig
from tinygent.memory.buffer_chat_memory import BufferChatMemoryConfig

T = TypeVar('T', bound='AbstractAgent')


class TinyBaseAgentConfig(AbstractAgentConfig[T], Generic[T]):
    """Configuration for BaseAgent."""

    type: Any = 'base'

    llm: AbstractLLMConfig
    tools: Sequence[AbstractToolConfig] = Field(default_factory=list)
    memory: AbstractMemoryConfig = Field(default_factory=BufferChatMemoryConfig)

    def build(self) -> T:
        """Build the BaseAgent instance from the configuration."""
        raise NotImplementedError('Subclasses must implement this method.')


class TinyBaseAgent(AbstractAgent):
    def __init__(
        self,
        llm: AbstractLLM,
        memory: AbstractMemory,
        tools: Sequence[AbstractTool] = (),
        **hooks_kwargs: Any,
    ) -> None:
        AgentHooks.__init__(self, **hooks_kwargs)

        self.llm = llm
        self.memory = memory

        self._tools = tools
        self._final_answer: str | None = None

    @property
    def tools(self) -> Sequence[AbstractTool]:
        return self._tools

    def get_tool(self, name: str) -> AbstractTool | None:
        return next((tool for tool in self.tools if tool.info.name == name), None)

    def get_tool_from_list(
        self, tools: list[AbstractTool], name: str
    ) -> AbstractTool | None:
        return next((tool for tool in tools if tool.info.name == name), None)

    def run_llm(
        self, run_id: str, fn: Callable, llm_input: TinyLLMInput, **kwargs
    ) -> Any:
        self.on_before_llm_call(run_id=run_id, llm_input=llm_input)
        try:
            result = fn(llm_input=llm_input, **kwargs)
            self.on_after_llm_call(run_id=run_id, llm_input=llm_input, result=result)
            return result
        except Exception as e:
            self.on_error(run_id=run_id, e=e)
            raise

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
        self.on_before_llm_call(run_id=run_id, llm_input=llm_input)
        try:
            result = fn(llm_input=llm_input, **kwargs)

            # if coroutine, await it to get async generator
            if isinstance(result, Awaitable):
                result = await result

            async for chunk in result:
                yield chunk

            self.on_after_llm_call(run_id=run_id, llm_input=llm_input, result=None)
        except Exception as e:
            self.on_error(run_id=run_id, e=e)
            raise

    def run_tool(
        self, run_id: str, tool: AbstractTool, call: TinyToolCall
    ) -> TinyToolResult:
        self.on_before_tool_call(run_id=run_id, tool=tool, args=call.arguments)
        try:
            result = tool(**call.arguments)
            call.metadata['executed'] = True
            call.result = result
            self.on_after_tool_call(
                run_id=run_id, tool=tool, args=call.arguments, result=result
            )

            tool_result = TinyToolResult(
                call_id=call.call_id or 'unknown',
                content=str(result),
            )
            tool_result.raw = tool
            return tool_result
        except Exception as e:
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

        return buf.getvalue()
