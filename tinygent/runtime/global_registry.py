from typing import Any
from typing import Callable

from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.tool import AbstractTool


class Registry:

    def __init__(self):

        self._registered_tools: dict[
            str,
            AbstractTool
        ] = {}

        self._registered_tool_convertors: dict[
            type[AbstractLLM],
            Callable[[AbstractTool], Any]
        ] = {}

    def register_tool(
        self,
        tool: AbstractTool
    ) -> None:

        if tool.info.name in self._registered_tools:
            raise ValueError(f'Tool {tool.info.name} already registered.')

        self._registered_tools[tool.info.name] = tool

    def register_tool_convertor(
        self,
        llm_type: type[AbstractLLM],
        fn: Callable[[AbstractTool], Any]
    ) -> None:

        if llm_type in self._registered_tool_convertors:
            raise ValueError(f'Convertor for {llm_type} already registered.')

        self._registered_tool_convertors[llm_type] = fn


class GlobalRegistry:

    _global_registry: Registry = Registry()

    @staticmethod
    def get_registry() -> Registry:
        return GlobalRegistry._global_registry
