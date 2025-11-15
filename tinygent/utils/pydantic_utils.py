from pydantic import BaseModel


def tiny_deep_copy(data: BaseModel) -> BaseModel:
    """Create a deep copy of a Pydantic model instance."""

    return type(data).model_validate(data.model_dump(mode='python', exclude_none=False))
