from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import TypeVar

from pydantic import ConfigDict

from tinygent.core.types.base import TinyModel

T = TypeVar('T')


class TinyModelBuildable(TinyModel, Generic[T], ABC):
    """Abstract base class for buildable TinyModel configurations."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')

    type: Any  # used as discriminator

    _discriminator_field: ClassVar[str] = 'type'

    @classmethod
    def get_discriminator_field(cls) -> str:
        """Get the name of the discriminator field."""
        return cls._discriminator_field

    @abstractmethod
    def build(self) -> Any:
        """Build the instance from the configuration."""
        pass

    @classmethod
    def rebuild_annotations(cls) -> bool:
        from tinygent.cli.builder import make_union
        from tinygent.core.runtime.global_registry import GlobalRegistry

        registry = GlobalRegistry.get_registry()
        should_rebuild = False

        mapping = {
            'llm': registry.get_llms,
            'embedders': registry.get_embedders,
            'crossencoders': registry.get_crossencoders,
            'tools': registry.get_tools,
            'memory_list': registry.get_memories,
            'agents': registry.get_agents,
        }

        for field_name, getter in mapping.items():
            if field_name not in cls.model_fields:
                continue

            union_ann = make_union(getter)  # type: ignore[arg-type]
            if cls.__annotations__.get(field_name) != union_ann:
                cls.__annotations__[field_name] = union_ann
                should_rebuild = True

        if should_rebuild:
            cls.model_rebuild(force=True)
            return True
        return False
