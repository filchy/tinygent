from typing import Sequence

from tinygent.datamodels.agent import AbstractAgent
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.tool import AbstractTool


class BaseAgent(AbstractAgent):
    def __init__(
        self,
        llm: AbstractLLM,
        tools: Sequence[AbstractTool] = (),
        memory_list: Sequence[AbstractMemory] = (),
    ) -> None:
        self.llm = llm
        self._tools = tools
        self.memory_list = memory_list

    @property
    def tools(self) -> Sequence[AbstractTool]:
        return self._tools

    def get_tool(self, name: str) -> AbstractTool | None:
        return next((tool for tool in self.tools if tool.info.name == name), None)

    def get_tool_from_list(
        self, tools: list[AbstractTool], name: str
    ) -> AbstractTool | None:
        return next((tool for tool in tools if tool.info.name == name), None)
