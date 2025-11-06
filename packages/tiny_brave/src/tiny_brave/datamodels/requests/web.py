from typing import Literal

from pydantic import Field

from tiny_brave.datamodels.requests.base import BaseSearchRequest


class WebSearchRequest(BaseSearchRequest):
    country: str | None = Field(
        default='US',
        description='2-character country code for where the results come from.',
    )

    ui_lang: str | None = Field(
        default='en-US', description='UI language preferred in response (e.g. en-US).'
    )

    count: int = Field(
        default=3,
        ge=1,
        le=20,
        description='Number of search results to return (max 20).',
    )

    offset: int = Field(
        default=0,
        ge=0,
        le=9,
        description='Zero-based page offset for pagination (max 9).',
    )

    safesearch: Literal['off', 'moderate', 'strict'] = Field(
        default='moderate', description='Adult content filter level.'
    )

    freshness: Literal['pd', 'pw', 'pm', 'py'] | None = Field(
        default=None,
        description=(
            'Time filter for results. '
            'Options: pd (24h), pw (7d), pm (31d), py (365d), '
            'or custom YYYY-MM-DDtoYYYY-MM-DD'
        ),
    )

    text_decorations: bool = Field(
        default=True, description='Whether snippets include decoration markers.'
    )

    result_filter: str | None = Field(
        default=None,
        description=(
            "Comma-delimited result types to include (e.g. 'discussions,videos,web')."
        ),
    )

    goggles_id: str | None = Field(
        default=None, description='Deprecated: use `goggles`. Goggles ID string.'
    )

    goggles: list[str] | None = Field(
        default=None, description='List of goggle URLs or definitions for re-ranking.'
    )

    units: Literal['metric', 'imperial'] | None = Field(
        default=None, description='Measurement system. Default inferred from country.'
    )

    extra_snippets: bool | None = Field(
        default=None,
        description='If true, return up to 5 additional alternative snippets.',
    )

    summary: bool | None = Field(
        default=None, description='If true, enables summary key generation in results.'
    )

    operators: bool = Field(
        default=True, description='Whether to apply search operators.'
    )
