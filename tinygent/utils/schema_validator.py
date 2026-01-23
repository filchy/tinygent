from typing import Any

from pydantic import ValidationError

from tinygent.core.types.base import TinyModel


def validate_schema(metadata: Any, schema: type[TinyModel]) -> TinyModel:
    try:
        return schema(**metadata)
    except ValidationError as e:
        raise e
