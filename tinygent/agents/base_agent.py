from typing import Any
from typing import Sequence

from tinygent.datamodels.agent import AbstractAgent
from tinygent.datamodels.agent_hooks import AgentHooks
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm_io import TinyLLMInput
from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.messages import TinyToolCall
from tinygent.datamodels.tool import AbstractTool


class BaseAgent(AbstractAgent, AgentHooks):
    def __init__(
        self,
        llm: AbstractLLM,
        tools: Sequence[AbstractTool] = (),
        memory_list: Sequence[AbstractMemory] = (),
    ) -> None:
        AgentHooks.__init__(self)

        self.llm = llm
        self.memory_list = memory_list

        self._tools = tools
        self._final_answer: str | None = None

    @property
    def final_answer(self) -> str | None:
        return self._final_answer

    @final_answer.setter
    def final_answer(self, value: str | None) -> None:
        self._final_answer = value

    @property
    def tools(self) -> Sequence[AbstractTool]:
        return self._tools

    def get_tool(self, name: str) -> AbstractTool | None:
        return next((tool for tool in self.tools if tool.info.name == name), None)

    def get_tool_from_list(
        self, tools: list[AbstractTool], name: str
    ) -> AbstractTool | None:
        return next((tool for tool in tools if tool.info.name == name), None)

    def run_llm(self, fn, llm_input: TinyLLMInput, **kwargs):
        self.on_before_llm_call(llm_input)
        try:
            result = fn(llm_input=llm_input, **kwargs)
            self.on_after_llm_call(llm_input, result)
            return result
        except Exception as e:
            self.on_error(e)
            raise

    def run_tool(self, tool: AbstractTool, call: TinyToolCall) -> Any:
        self.on_before_tool_call(tool, call.arguments)
        try:
            result = tool(**call.arguments)
            call.metadata['executed'] = True
            call.result = result
            self.on_after_tool_call(tool, call.arguments, result)
            return result
        except Exception as e:
            self.on_error(e)
            raise
