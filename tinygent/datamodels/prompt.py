import logging
from typing import ClassVar

from pydantic import model_validator

from tinygent.types.base import TinyModel
from tinygent.utils.jinja_utils import validate_template


class TinyPromptTemplate(TinyModel):
    _template_fields: ClassVar[dict[str, set[str]]] = {}

    @model_validator(mode='after')
    def _validate_template_fields(self) -> 'TinyPromptTemplate':
        logger = logging.getLogger(__name__)

        for field_name, required in self._template_fields.items():
            logger.debug(
                f'Validating prompt template field: {field_name} with required fields: {required}'
            )
            value = getattr(self, field_name, None)
            if value is None:
                raise ValueError(
                    f'Field "{field_name}" is required in the prompt template.'
                )
            if not validate_template(value, required_fields=required):
                raise ValueError(
                    f'{self.__class__.__name__}.{field_name} is missing required fields: {required}'
                )
        return self
