from __future__ import annotations

from typing import Any
from typing import Generic
from typing import Sequence
from typing import TypeVar

from pydantic import Field

from tinygent.datamodels.agent import AbstractAgent
from tinygent.datamodels.agent import AbstractAgentConfig
from tinygent.datamodels.agent_hooks import AgentHooks
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm import AbstractLLMConfig
from tinygent.datamodels.llm_io_input import TinyLLMInput
from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.messages import TinyToolCall
from tinygent.datamodels.messages import TinyToolResult
from tinygent.datamodels.tool import AbstractTool
from tinygent.tools.tool import ToolConfig

T = TypeVar('T', bound='AbstractAgent')


class TinyBaseAgentConfig(AbstractAgentConfig[T], Generic[T]):
    """Configuration for BaseAgent."""

    type: Any = 'base'

    llm: AbstractLLMConfig
    tools: Sequence[ToolConfig] = Field(default_factory=list)
    memory_list: Sequence[AbstractMemory] = Field(default_factory=list)

    def build(self) -> T:
        """Build the BaseAgent instance from the configuration."""
        raise NotImplementedError('Subclasses must implement this method.')


class TinyBaseAgent(AbstractAgent, AgentHooks):
    def __init__(
        self,
        llm: AbstractLLM,
        tools: Sequence[AbstractTool] = (),
        memory_list: Sequence[AbstractMemory] = (),
        **hooks_kwargs: Any,
    ) -> None:
        AgentHooks.__init__(self, **hooks_kwargs)

        self.llm = llm
        self.memory_list = memory_list

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

    def run_llm(self, fn, llm_input: TinyLLMInput, **kwargs) -> Any:
        self.on_before_llm_call(llm_input)
        try:
            result = fn(llm_input=llm_input, **kwargs)
            self.on_after_llm_call(llm_input, result)
            return result
        except Exception as e:
            self.on_error(e)
            raise

    def run_tool(self, tool: AbstractTool, call: TinyToolCall) -> TinyToolResult:
        self.on_before_tool_call(tool, call.arguments)
        try:
            result = tool(**call.arguments)
            call.metadata['executed'] = True
            call.result = result
            self.on_after_tool_call(tool, call.arguments, result)

            return TinyToolResult(
                call_id=call.call_id or 'unknown',
                content=str(result),
            )
        except Exception as e:
            self.on_error(e)
            raise
