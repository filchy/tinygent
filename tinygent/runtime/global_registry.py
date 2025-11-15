from __future__ import annotations

import logging
import typing

if typing.TYPE_CHECKING:
    from tinygent.datamodels.agent import AbstractAgent
    from tinygent.datamodels.agent import AbstractAgentConfig
    from tinygent.datamodels.llm import AbstractLLM
    from tinygent.datamodels.llm import AbstractLLMConfig
    from tinygent.datamodels.memory import AbstractMemory
    from tinygent.datamodels.memory import AbstractMemoryConfig
    from tinygent.datamodels.tool import AbstractTool
    from tinygent.datamodels.tool import AbstractToolConfig

logger = logging.getLogger(__name__)


class Registry:
    def __init__(self) -> None:
        # agents
        self._registered_agents: dict[
            str, tuple[type[AbstractAgentConfig], type[AbstractAgent]]
        ] = {}

        # llms
        self._registered_llms: dict[
            str, tuple[type[AbstractLLMConfig], type[AbstractLLM]]
        ] = {}

        # memories
        self._registered_memories: dict[
            str, tuple[type[AbstractMemoryConfig], type[AbstractMemory]]
        ] = {}

        # tools
        self._registered_tools: dict[
            str, tuple[type[AbstractToolConfig], type[AbstractTool]]
        ] = {}

    def _rebuild_annotations(self) -> None:
        from tinygent.types import TinyModelBuildable

        configs: list[type[TinyModelBuildable]] = []
        configs.extend(cfg for cfg, _ in self._registered_agents.values())
        configs.extend(cfg for cfg, _ in self._registered_llms.values())
        configs.extend(cfg for cfg, _ in self._registered_memories.values())
        configs.extend(cfg for cfg, _ in self._registered_tools.values())

        for config_cls in configs:
            if issubclass(config_cls, TinyModelBuildable):
                config_cls.rebuild_annotations()

    def _registration_changed(self) -> None:
        logger.debug('Registry changed, rebuilding annotations')
        self._rebuild_annotations()

    # agents
    def register_agent(
        self,
        name: str,
        config_class: type[AbstractAgentConfig],
        agent_class: type[AbstractAgent],
    ) -> None:
        logger.debug(f'Registering agent {name}')
        if name in self._registered_agents:
            raise ValueError(f'Agent {name} already registered.')

        self._registered_agents[name] = (config_class, agent_class)
        self._registration_changed()

    def get_agent(
        self, name: str
    ) -> tuple[type[AbstractAgentConfig], type[AbstractAgent]]:
        logger.debug(f'Getting agent {name}')
        if name not in self._registered_agents:
            raise ValueError(f'Agent {name} not registered.')

        return self._registered_agents[name]

    def get_agents(
        self,
    ) -> dict[str, tuple[type[AbstractAgentConfig], type[AbstractAgent]]]:
        logger.debug('Getting all registered agents')
        return self._registered_agents

    # llms
    def register_llm(
        self,
        name: str,
        config_class: type[AbstractLLMConfig],
        llm_class: type[AbstractLLM],
    ) -> None:
        logger.debug(f'Registering LLM {name}')
        if name in self._registered_llms:
            raise ValueError(f'LLM {name} already registered.')

        self._registered_llms[name] = (config_class, llm_class)
        self._registration_changed()

    def get_llm(self, name: str) -> tuple[type[AbstractLLMConfig], type[AbstractLLM]]:
        logger.debug(f'Getting LLM {name}')
        if name not in self._registered_llms:
            raise ValueError(f'LLM {name} not registered.')

        return self._registered_llms[name]

    def get_llms(self) -> dict[str, tuple[type[AbstractLLMConfig], type[AbstractLLM]]]:
        logger.debug('Getting all registered LLMs')
        return self._registered_llms

    # memories
    def register_memory(
        self,
        name: str,
        config_class: type[AbstractMemoryConfig],
        memory_class: type[AbstractMemory],
    ) -> None:
        logger.debug(f'Registering memory {name}')
        if name in self._registered_memories:
            raise ValueError(f'Memory {name} already registered.')

        self._registered_memories[name] = (config_class, memory_class)
        self._registration_changed()

    def get_memory(
        self, name: str
    ) -> tuple[type[AbstractMemoryConfig], type[AbstractMemory]]:
        logger.debug(f'Getting memory {name}')
        if name not in self._registered_memories:
            raise ValueError(f'Memory {name} not registered.')

        return self._registered_memories[name]

    def get_memories(
        self,
    ) -> dict[str, tuple[type[AbstractMemoryConfig], type[AbstractMemory]]]:
        logger.debug('Getting all registered memories')
        return self._registered_memories

    # tools
    def register_tool(
        self,
        name: str,
        config_class: type[AbstractToolConfig],
        tool_class: type[AbstractTool],
    ) -> None:
        logger.debug(f'Registering tool {name}')
        if name in self._registered_tools:
            raise ValueError(f'Tool {name} already registered.')

        self._registered_tools[name] = (config_class, tool_class)
        self._registration_changed()

    def get_tool(self, name: str) -> tuple[type[AbstractToolConfig], type[AbstractTool]]:
        logger.debug(f'Getting tool {name}')
        if name not in self._registered_tools:
            raise ValueError(f'Tool {name} not registered.')

        return self._registered_tools[name]

    def get_tools(
        self,
    ) -> dict[str, tuple[type[AbstractToolConfig], type[AbstractTool]]]:
        logger.debug('Getting all registered tools')
        return self._registered_tools


class GlobalRegistry:
    _global_registry: Registry = Registry()

    @staticmethod
    def get_registry() -> Registry:
        return GlobalRegistry._global_registry
