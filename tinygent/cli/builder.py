from typing import Annotated
from typing import Any
from typing import Callable
from typing import Mapping
from typing import TypeVar
from typing import Union

from pydantic import Field

from tinygent.core.types.discriminator import HasDiscriminatorField

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
