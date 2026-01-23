from typing import Annotated
from typing import Any
from typing import Callable
from typing import Mapping
from typing import TypeVar
from typing import Union

from pydantic import Field
from pydantic import TypeAdapter

from tinygent.core.types.base import TinyModel
from tinygent.core.types.discriminator import HasDiscriminatorField

T = TypeVar('T', bound=HasDiscriminatorField)

_discovered_modules: bool = False


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


def parse_config(
    config: dict | TinyModel,
    getter: Callable[[], Mapping[str, tuple[type[T], Any]]],
) -> T:
    """Generic parser: returns the validated config model instance."""
    if isinstance(config, TinyModel):
        config = config.model_dump()

    ConfigUnion = make_union(getter)
    adapter = TypeAdapter(ConfigUnion)
    return adapter.validate_python(config)


def parse_model(model: str, model_provider: str | None = None) -> tuple[str, str]:
    """Parse model and its provider."""
    if ':' not in model and model_provider is None:
        raise ValueError(
            'Model string must be in the format "model_provider:model_name" '
            'or a model_provider must be specified.'
        )

    if model_provider is None:
        model_provider, model = model.split(':')

    return model_provider, model


def check_modules() -> None:
    """Check if modules were already discovered or not. If not discovers them."""
    if _discovered_modules:
        return

    from tinygent.cli.utils import discover_and_register_components

    discover_and_register_components()
