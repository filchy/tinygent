from __future__ import annotations

import logging
import typing

if typing.TYPE_CHECKING:
    from tinygent.core.datamodels.agent import AbstractAgent
    from tinygent.core.datamodels.agent import AbstractAgentConfig
    from tinygent.core.datamodels.cross_encoder import AbstractCrossEncoder
    from tinygent.core.datamodels.cross_encoder import AbstractCrossEncoderConfig
    from tinygent.core.datamodels.embedder import AbstractEmbedder
    from tinygent.core.datamodels.embedder import AbstractEmbedderConfig
    from tinygent.core.datamodels.llm import AbstractLLM
    from tinygent.core.datamodels.llm import AbstractLLMConfig
    from tinygent.core.datamodels.memory import AbstractMemory
    from tinygent.core.datamodels.memory import AbstractMemoryConfig
    from tinygent.core.datamodels.middleware import AbstractMiddleware
    from tinygent.core.datamodels.middleware import AbstractMiddlewareConfig
    from tinygent.core.datamodels.tool import AbstractTool
    from tinygent.core.datamodels.tool import AbstractToolConfig

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

        # embedders
        self._registered_embedders: dict[
            str, tuple[type[AbstractEmbedderConfig], type[AbstractEmbedder]]
        ] = {}

        # cross-encoders
        self._registered_crossencoders: dict[
            str, tuple[type[AbstractCrossEncoderConfig], type[AbstractCrossEncoder]]
        ] = {}

        # memories
        self._registered_memories: dict[
            str, tuple[type[AbstractMemoryConfig], type[AbstractMemory]]
        ] = {}

        # tools
        self._registered_tools: dict[
            str, tuple[type[AbstractToolConfig], type[AbstractTool]]
        ] = {}

        # middlewares
        self._registered_middlewares: dict[
            str, tuple[type[AbstractMiddlewareConfig], type[AbstractMiddleware]]
        ] = {}

    def _rebuild_annotations(self) -> None:
        from tinygent.core.types.builder import TinyModelBuildable

        configs: list[type[TinyModelBuildable]] = []
        configs.extend(cfg for cfg, _ in self._registered_agents.values())
        configs.extend(cfg for cfg, _ in self._registered_llms.values())
        configs.extend(cfg for cfg, _ in self._registered_embedders.values())
        configs.extend(cfg for cfg, _ in self._registered_crossencoders.values())
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
        logger.debug('Registering agent %s', name)
        if name in self._registered_agents:
            raise ValueError(f'Agent {name} already registered.')

        self._registered_agents[name] = (config_class, agent_class)
        self._registration_changed()

    def get_agent(
        self, name: str
    ) -> tuple[type[AbstractAgentConfig], type[AbstractAgent]]:
        logger.debug('Getting agent %s', name)
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
        logger.debug('Registering LLM %s', name)
        if name in self._registered_llms:
            raise ValueError(f'LLM {name} already registered.')

        self._registered_llms[name] = (config_class, llm_class)
        self._registration_changed()

    def get_llm(self, name: str) -> tuple[type[AbstractLLMConfig], type[AbstractLLM]]:
        logger.debug('Getting LLM %s', name)
        if name not in self._registered_llms:
            raise ValueError(f'LLM {name} not registered.')

        return self._registered_llms[name]

    def get_llms(self) -> dict[str, tuple[type[AbstractLLMConfig], type[AbstractLLM]]]:
        logger.debug('Getting all registered LLMs')
        return self._registered_llms

    # embedders
    def register_embedder(
        self,
        name: str,
        config_class: type[AbstractEmbedderConfig],
        embedder_class: type[AbstractEmbedder],
    ) -> None:
        logger.debug('Registering Embedder %s', name)
        if name in self._registered_embedders:
            raise ValueError(f'Embedder {name} already registered.')

        self._registered_embedders[name] = (config_class, embedder_class)
        self._registration_changed()

    def get_embedder(
        self, name: str
    ) -> tuple[type[AbstractEmbedderConfig], type[AbstractEmbedder]]:
        logger.debug('Getting Embedder %s', name)
        if name not in self._registered_embedders:
            raise ValueError(f'Embedder {name} not registered.')

        return self._registered_embedders[name]

    def get_embedders(
        self,
    ) -> dict[str, tuple[type[AbstractEmbedderConfig], type[AbstractEmbedder]]]:
        logger.debug('Getting all registered Embedders.')
        return self._registered_embedders

    # cross-encoders
    def register_crossencoder(
        self,
        name: str,
        config_class: type[AbstractCrossEncoderConfig],
        crossencoder_class: type[AbstractCrossEncoder],
    ) -> None:
        logger.debug('Registering cross-encoder %s', name)
        if name in self._registered_crossencoders:
            raise ValueError(f'Cross-encoder {name} already registered.')

        self._registered_crossencoders[name] = (config_class, crossencoder_class)
        self._registration_changed()

    def get_crossencoder(
        self, name: str
    ) -> tuple[type[AbstractCrossEncoderConfig], type[AbstractCrossEncoder]]:
        logger.debug('Getting cross-encoder %s', name)
        if name not in self._registered_crossencoders:
            raise ValueError(f'Cross-encoder {name} not registered.')

        return self._registered_crossencoders[name]

    def get_crossencoders(
        self,
    ) -> dict[str, tuple[type[AbstractCrossEncoderConfig], type[AbstractCrossEncoder]]]:
        logger.debug('Gettings all registered cross-encoders')
        return self._registered_crossencoders

    # memories
    def register_memory(
        self,
        name: str,
        config_class: type[AbstractMemoryConfig],
        memory_class: type[AbstractMemory],
    ) -> None:
        logger.debug('Registering memory %s', name)
        if name in self._registered_memories:
            raise ValueError(f'Memory {name} already registered.')

        self._registered_memories[name] = (config_class, memory_class)
        self._registration_changed()

    def get_memory(
        self, name: str
    ) -> tuple[type[AbstractMemoryConfig], type[AbstractMemory]]:
        logger.debug('Getting memory %s', name)
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
        logger.debug('Registering tool %s', name)
        if name in self._registered_tools:
            raise ValueError(f'Tool {name} already registered.')

        self._registered_tools[name] = (config_class, tool_class)
        self._registration_changed()

    def get_tool(self, name: str) -> tuple[type[AbstractToolConfig], type[AbstractTool]]:
        logger.debug('Getting tool %s', name)
        if name not in self._registered_tools:
            raise ValueError(f'Tool {name} not registered.')

        return self._registered_tools[name]

    def get_tools(
        self,
    ) -> dict[str, tuple[type[AbstractToolConfig], type[AbstractTool]]]:
        logger.debug('Getting all registered tools')
        return self._registered_tools

    # middlewares
    def register_middleware(
        self,
        name: str,
        config_class: type[AbstractMiddlewareConfig],
        middleware_class: type[AbstractMiddleware],
    ) -> None:
        logger.debug('Registering middleware %s', name)
        if name in self._registered_middlewares:
            raise ValueError(f'Middleware {name} already registered.')

        self._registered_middlewares[name] = (config_class, middleware_class)
        self._registration_changed()

    def get_middleware(
        self, name: str
    ) -> tuple[type[AbstractMiddlewareConfig], type[AbstractMiddleware]]:
        logger.debug('Getting middleware %s', name)
        if name not in self._registered_middlewares:
            raise ValueError(f'Middleware {name} not registered.')

        return self._registered_middlewares[name]

    def get_middlewares(
        self,
    ) -> dict[str, tuple[type[AbstractMiddlewareConfig], type[AbstractMiddleware]]]:
        logger.debug('Getting all registered middlewares.')
        return self._registered_middlewares


class GlobalRegistry:
    _global_registry: Registry = Registry()

    @staticmethod
    def get_registry() -> Registry:
        return GlobalRegistry._global_registry
