from typing import Any
from typing import Callable

from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.tool import AbstractTool


class Registry:

    def __init__(self):

        self._registered_llms: dict[
            str,
            AbstractLLM
        ] = {}

        self._registered_tools: dict[
            str,
            AbstractTool
        ] = {}

    # llms
    def register_llm(
        self,
        name: str,
        llm: AbstractLLM
    ) -> None:

        if name in self._registered_llms:
            raise ValueError(f'LLM {name} already registered.')

        self._registered_llms[name] = llm

    def get_llm(
        self,
        name: str
    ) -> AbstractLLM:

        if name not in self._registered_llms:
            raise ValueError(f'LLM {name} not registered.')

        return self._registered_llms[name]

    def get_llms(self) -> dict[str, AbstractLLM]:

        return self._registered_llms

    # tools
    def register_tool(
        self,
        tool: AbstractTool
    ) -> None:

        if tool.info.name in self._registered_tools:
            raise ValueError(f'Tool {tool.info.name} already registered.')

        self._registered_tools[tool.info.name] = tool

    def get_tool(
        self,
        name: str
    ) -> AbstractTool:
        
        if name not in self._registered_tools:
            raise ValueError(f'Tool {name} not registered.')

        return self._registered_tools[name]

    def get_tools(self) -> dict[str, AbstractTool]:

        return self._registered_tools


class GlobalRegistry:

    _global_registry: Registry = Registry()

    @staticmethod
    def get_registry() -> Registry:
        return GlobalRegistry._global_registry
