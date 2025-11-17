from typing import Annotated
from typing import Any
from typing import Callable
from typing import Mapping
from typing import TypeVar
from typing import Union

from pydantic import Field
from pydantic import TypeAdapter

from tinygent.datamodels.agent import AbstractAgent
from tinygent.datamodels.agent import AbstractAgentConfig
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm import AbstractLLMConfig
from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.memory import AbstractMemoryConfig
from tinygent.datamodels.tool import AbstractTool
from tinygent.datamodels.tool import AbstractToolConfig
from tinygent.runtime.global_registry import GlobalRegistry
from tinygent.types.discriminator import HasDiscriminatorField
from tinygent.types.base import TinyModel

T = TypeVar('T', bound=HasDiscriminatorField)


def make_union(getter: Callable[[], Mapping[str, tuple[type[T], Any]]]):
    """Create a discriminated union type from registered config classes."""
    mapping = getter()
    config_classes = [cfg for cfg, _ in mapping.values()]

    if not config_classes:
        return None

    first = config_classes[0].get_discriminator_field()
    if not all(cfg.get_discriminator_field() == first for cfg in config_classes):
        raise ValueError('Inconsistent discriminator fields.')

    return Annotated[Union[tuple(config_classes)], Field(discriminator=first)]


def _parse_config(
    config: dict | TinyModel,
    getter: Callable[[], Mapping[str, tuple[type[T], Any]]],
) -> T:
    """Generic parser: returns the validated config model instance."""
    if isinstance(config, TinyModel):
        config = config.model_dump()

    ConfigUnion = make_union(getter)
    adapter = TypeAdapter(ConfigUnion)
    return adapter.validate_python(config)


def build_agent(config: dict | AbstractAgentConfig) -> AbstractAgent:
    if isinstance(config, AbstractAgentConfig):
        config = config.model_dump()

    agent_config = _parse_config(
        config, lambda: GlobalRegistry.get_registry().get_agents()
    )
    return agent_config.build()


def build_llm(config: dict | AbstractLLMConfig) -> AbstractLLM:
    if isinstance(config, AbstractLLMConfig):
        config = config.model_dump()

    llm_config = _parse_config(config, lambda: GlobalRegistry.get_registry().get_llms())
    return llm_config.build()


def build_memory(config: dict | AbstractMemoryConfig) -> AbstractMemory:
    if isinstance(config, AbstractMemoryConfig):
        config = config.model_dump()

    memory_config = _parse_config(
        config, lambda: GlobalRegistry.get_registry().get_memories()
    )
    return memory_config.build()


def build_tool(config: dict | AbstractToolConfig) -> AbstractTool:
    if isinstance(config, AbstractToolConfig):
        config = config.model_dump()

    tool_config = _parse_config(
        config, lambda: GlobalRegistry.get_registry().get_tools()
    )
    return tool_config.build()
