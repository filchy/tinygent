from pydantic import Field

from tiny_brave.datamodels.requests.base import BaseSearchRequest


class VideoSearchRequest(BaseSearchRequest):
    ui_lang: str | None = Field(
        default='en-US',
        description=(
            'User interface language preferred in response. '
            'Format <language_code>-<country_code>.'
        ),
    )

    count: int = Field(
        default=20,
        ge=1,
        le=50,
        description=('The number of search results returned. Maximum is 50.'),
    )

    offset: int = Field(
        default=0,
        ge=0,
        le=9,
        description=('Zero-based offset for pagination. Maximum is 9.'),
    )

    safesearch: str | None = Field(
        default='moderate', description=('Filter adult content: off, moderate, strict.')
    )

    freshness: str | None = Field(
        default=None,
        description=(
            'Filter results by discovery time. '
            'Supported: pd, pw, pm, py, or YYYY-MM-DDtoYYYY-MM-DD.'
        ),
    )

    operators: bool = Field(
        default=True, description='Whether to apply search operators.'
    )
