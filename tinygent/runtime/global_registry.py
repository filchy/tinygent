from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm import AbstractLLM
    from tinygent.datamodels.tool import AbstractTool


class Registry:
    def __init__(self) -> None:
        self._registered_llms: dict[str, AbstractLLM] = {}

        self._registered_tools: dict[str, AbstractTool] = {}
        self._registered_hidden_tools: dict[str, AbstractTool] = {}

    # llms
    def register_llm(self, name: str, llm: AbstractLLM) -> None:
        if name in self._registered_llms:
            raise ValueError(f'LLM {name} already registered.')

        self._registered_llms[name] = llm

    def get_llm(self, name: str) -> AbstractLLM:
        if name not in self._registered_llms:
            raise ValueError(f'LLM {name} not registered.')

        return self._registered_llms[name]

    def get_llms(self) -> dict[str, AbstractLLM]:
        return self._registered_llms

    # tools
    def register_tool(self, tool: AbstractTool, hidden: bool = False) -> None:
        if tool.info.name in self.get_tools(include_hidden=True):
            raise ValueError(f'Tool {tool.info.name} already registered.')

        if hidden:
            self._registered_hidden_tools[tool.info.name] = tool
        else:
            self._registered_tools[tool.info.name] = tool

    def get_tool(self, name: str) -> AbstractTool:
        if name in self._registered_tools:
            return self._registered_tools[name]
        if name in self._registered_hidden_tools:
            return self._registered_hidden_tools[name]
        raise ValueError(f'Tool {name} not registered.')

    def get_tools(self, include_hidden: bool = False) -> dict[str, AbstractTool]:
        if include_hidden:
            return {**self._registered_tools, **self._registered_hidden_tools}
        return self._registered_tools


class GlobalRegistry:
    _global_registry: Registry = Registry()

    @staticmethod
    def get_registry() -> Registry:
        return GlobalRegistry._global_registry
