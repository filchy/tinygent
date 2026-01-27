import logging
from typing import Any
from typing import ClassVar

from pydantic import ConfigDict
from pydantic import model_validator

from tinygent.core.types.base import TinyModel
from tinygent.utils.jinja_utils import validate_template


class TinyPrompt(TinyModel):
    """Base class for prompt templates with template field validation."""

    class UserSystem(TinyModel):
        """User & System prompt template"""

        user: str
        system: str

    model_config = ConfigDict(frozen=True)

    _template_fields: ClassVar[dict[str, set[str]]] = {}

    @model_validator(mode='after')
    def _validate_template_fields(self) -> 'TinyPrompt':
        logger = logging.getLogger(__name__)

        for field_path, required in self._template_fields.items():
            logger.debug(
                f'Validating prompt template field: {field_path} with required fields: {required}'
            )

            parts = field_path.split('.')
            value: TinyPrompt | Any | None = self

            for part in parts:
                value = getattr(value, part, None)

                if value is None:
                    raise ValueError(
                        f'Field "{field_path}" is required in the prompt template.'
                    )

            if not isinstance(value, str):
                raise ValueError(
                    f'Field "{field_path}" resolved to non-string value: {type(value)}. '
                    f'Template validation requires a string.'
                )

            if not validate_template(value, required_fields=required):
                raise ValueError(
                    f'{self.__class__.__name__}.{field_path} is missing required fields: {required}'
                )
        return self
