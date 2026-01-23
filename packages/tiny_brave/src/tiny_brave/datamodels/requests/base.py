from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from pydantic import create_model
from pydantic import field_validator
from pydantic import model_validator

from tiny_brave.constants import MAX_QUERY_LENGTH
from tiny_brave.constants import MAX_QUERY_TERMS
from tiny_brave.exceptions import TinyBraveClientError
from tinygent.core.types.base import TinyModel


class BaseSearchRequest(TinyModel):
    q: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description=(
            f'The search query string used to find relevant news articles. '
            f'Maximum length is {MAX_QUERY_LENGTH} characters.'
        ),
    )

    country: str | None = Field(
        default='US',
        description='2-character country code for where the results come from.',
    )

    spellcheck: bool = Field(
        default=False,
        description='Whether to enable spellchecking for the search query.',
    )

    search_lang: str | None = Field(
        default=None,
        description=(
            'The search language preference. The 2 or more character language code for '
            'which the search results are provided.'
        ),
    )

    @field_validator('q')
    def validate_query(cls, value: str) -> str:
        if len(value.strip().split()) > MAX_QUERY_TERMS:
            raise TinyBraveClientError(
                f'Query exceeds maximum term limit of {MAX_QUERY_TERMS}.'
            )
        return value

    @model_validator(mode='before')
    def sanitize(cls, values: dict) -> dict:
        for name, field in cls.model_fields.items():
            if name not in values:
                continue
            v = values[name]
            if v is None:
                continue

            Tmp = create_model(  # type: ignore
                f'_Tmp_{cls.__name__}_{name}',
                __base__=BaseModel,
                **{name: (field.annotation, field)},  # type: ignore
            )

            try:
                validated = Tmp(**{name: v})
                values[name] = getattr(validated, name)
            except ValidationError:
                values[name] = field.default

        return values
