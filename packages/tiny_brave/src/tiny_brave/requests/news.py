from pydantic import Field

from tiny_brave.requests.base import BaseSearchRequest


class NewsSearchRequest(BaseSearchRequest):
    count: int = Field(
        default=1,
        ge=1,
        le=20,
        description='The maximum number of news articles to return (1-20).'
    )

    class Config:
        populate_by_name = True
