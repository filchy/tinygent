from typing import Any
from pydantic import BaseModel
from pydantic import ValidationError


def validate_schema(metadata: Any, schema: type[BaseModel]) -> BaseModel:

    try:
        return schema(**metadata)
    except ValidationError as e:

        raise e
