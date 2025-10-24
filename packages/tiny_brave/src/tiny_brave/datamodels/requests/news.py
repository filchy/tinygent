import re
from typing import Literal

from pydantic import Field
from pydantic import field_validator

from tiny_brave.datamodels.requests.base import BaseSearchRequest
from tiny_brave.exceptions import TinyBraveClientError


class NewsSearchRequest(BaseSearchRequest):
    ui_lang: str | None = Field(
        default='en-US', description='UI language preferred in response (e.g. en-US).'
    )

    count: int = Field(
        default=3,
        ge=1,
        le=50,
        description='The number of search results to return (1–50).',
    )

    offset: int = Field(
        default=0, le=9, description='Zero-based page offset for pagination (max 9).'
    )

    safesearch: Literal['off', 'moderate', 'strict'] = Field(
        default='moderate', description='Adult content filter level.'
    )

    freshness: str | None = Field(
        default=None,
        description=(
            'Time filter for results. '
            'Options: pd (24h), pw (7d), pm (31d), py (365d), '
            'or custom YYYY-MM-DDtoYYYY-MM-DD'
        ),
    )

    extra_snippets: bool | None = Field(
        default=None,
        description='If true, return up to 5 additional alternative snippets.',
    )

    goggles: list[str] | None = Field(
        default=None, description='List of goggle URLs or definitions for re-ranking.'
    )

    operators: bool = Field(
        default=True, description='Whether to apply search operators.'
    )

    @field_validator('country')
    def validate_country(cls, v: str | None) -> str | None:
        if v is not None and len(v) != 2:
            raise TinyBraveClientError('Country code must be exactly 2 characters.')
        return v

    @field_validator('ui_lang')
    def validate_ui_lang(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # must be "xx" or "xx-XX"
        if not re.match(r'^[a-z]{2}(-[A-Z]{2})?$', v):
            raise TinyBraveClientError(
                f'Invalid ui_lang "{v}". Must be like "en" or "en-US".'
            )
        return v

    @field_validator('freshness')
    def validate_freshness(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if v in ('pd', 'pw', 'pm', 'py'):
            return v
        if re.match(r'^\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2}$', v):
            return v
        raise TinyBraveClientError(
            f'Invalid freshness "{v}". Must be one of pd, pw, pm, py, '
            f'or date range YYYY-MM-DDtoYYYY-MM-DD.'
        )
