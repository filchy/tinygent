from typing import overload

from tinygent.core.datamodels.memory import AbstractMemory
from tinygent.core.datamodels.memory import AbstractMemoryConfig
from tinygent.core.factory.helper import check_modules
from tinygent.core.factory.helper import parse_config
from tinygent.core.runtime.global_registry import GlobalRegistry


@overload
def build_memory(memory: dict | AbstractMemoryConfig) -> AbstractMemory: ...


@overload
def build_memory(memory: str, **kwargs) -> AbstractMemory: ...


def build_memory(memory: dict | AbstractMemoryConfig | str, **kwargs) -> AbstractMemory:
    """Build tiny memory."""
    check_modules()

    if isinstance(memory, str):
        memory = {'type': memory, **kwargs}

    if isinstance(memory, AbstractMemoryConfig):
        memory = memory.model_dump()

    memory_config = parse_config(
        memory, lambda: GlobalRegistry.get_registry().get_memories()
    )
    return memory_config.build()
