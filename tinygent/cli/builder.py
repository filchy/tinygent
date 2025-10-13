from typing import Annotated
from typing import Callable
from typing import Mapping
from typing import TypeVar
from typing import Union

from pydantic import Field
from pydantic import TypeAdapter

from tinygent.cli.utils import discover_and_register_components
from tinygent.datamodels.agent import AbstractAgentConfig
from tinygent.datamodels.llm import AbstractLLMConfig
from tinygent.datamodels.memory import AbstractMemoryConfig
from tinygent.runtime.global_registry import GlobalRegistry
from tinygent.runtime.global_registry import Registry
from tinygent.types.base import TinyModel
from tinygent.types.discriminator import HasDiscriminatorField

T = TypeVar('T', bound=HasDiscriminatorField)


def _make_union(getter: Callable[[Registry], Mapping[str, tuple[type[T], object]]]):
    registry = GlobalRegistry.get_registry()
    config_classes = [cfg for cfg, _ in getter(registry).values()]

    if not config_classes:
        raise ValueError('No configurations registered.')

    first = config_classes[0].get_discriminator_field()
    if not all(cfg.get_discriminator_field() == first for cfg in config_classes):
        raise ValueError('Inconsistent discriminator fields.')

    return Annotated[Union[tuple(config_classes)], Field(discriminator=first)]


def _parse_config(
    config: dict | TinyModel,
    getter: Callable[[Registry], Mapping[str, tuple[type[T], object]]],
) -> T:
    """
    Generic parser: returns the validated config model instance.
    """
    if isinstance(config, TinyModel):
        config = config.model_dump()

    discover_and_register_components()
    ConfigUnion = _make_union(getter)
    adapter = TypeAdapter(ConfigUnion)
    return adapter.validate_python(config)


def build_agent(config: dict | AbstractAgentConfig):
    if isinstance(config, AbstractAgentConfig):
        config = config.model_dump()

    agent_config = _parse_config(config, lambda r: r.get_agents())
    return agent_config.build()


def build_llm(config: dict | AbstractLLMConfig):
    if isinstance(config, AbstractLLMConfig):
        config = config.model_dump()

    llm_config = _parse_config(config, lambda r: r.get_llms())
    return llm_config.build()


def build_memory(config: dict | AbstractMemoryConfig):
    if isinstance(config, AbstractMemoryConfig):
        config = config.model_dump()

    memory_config = _parse_config(config, lambda r: r.get_memories())
    return memory_config.build()
