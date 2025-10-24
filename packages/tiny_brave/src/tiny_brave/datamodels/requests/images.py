from typing import Literal

from pydantic import Field

from tiny_brave.datamodels.requests.base import BaseSearchRequest


class ImagesSearchReuest(BaseSearchRequest):
    count: int = Field(
        default=3,
        ge=1,
        le=200,
        description='The number of search results to return (1–200).',
    )

    safesearch: Literal['off', 'strict'] | None = Field(
        default=None,
        description=(
            'Adult content filter level. Options are "off" or "strict". '
            'off - No filtering is done. '
            'strict - Drops all adult content from search results.'
        ),
    )
