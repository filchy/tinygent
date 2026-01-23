from typing import List
from typing import Literal
from typing import Optional

from pydantic import Field

from tinygent.core.types.base import TinyModel


class Thumbnail(TinyModel):
    """Aggregated details representing the news thumbnail."""

    src: str = Field(
        ...,
        description='The served URL of the thumbnail associated with the news article.',
    )
    original: Optional[str] = Field(
        None,
        description='The original URL of the thumbnail associated with the news article.',
    )


class MetaUrl(TinyModel):
    """Aggregated information about a URL."""

    scheme: Optional[str] = Field(
        None, description='The protocol scheme extracted from the URL.'
    )
    netloc: Optional[str] = Field(
        None, description='The network location part extracted from the URL.'
    )
    hostname: Optional[str] = Field(
        None, description='The lowercased domain name extracted from the URL.'
    )
    favicon: Optional[str] = Field(None, description='The favicon used for the URL.')
    path: Optional[str] = Field(
        None, description='The hierarchical path of the URL useful as a display string.'
    )


class NewsResult(TinyModel):
    """A model representing a news result for the requested query."""

    type: str = Field(
        ...,
        description='The type of news search API result. The value is always news_result.',
    )
    url: str = Field(..., description='The source URL of the news article.')
    title: str = Field(..., description='The title of the news article.')
    description: Optional[str] = Field(
        None, description='The description for the news article.'
    )
    age: Optional[str] = Field(
        None, description='A human readable representation of the page age.'
    )
    page_age: Optional[str] = Field(
        None, description='The page age found from the source web page.'
    )
    page_fetched: Optional[str] = Field(
        None,
        description='The ISO date time when the page was last fetched. Format: YYYY-MM-DDTHH:MM:SSZ.',
    )
    breaking: Optional[bool] = Field(
        None, description='Whether the result includes breaking news.'
    )
    thumbnail: Optional[Thumbnail] = Field(
        None, description='The thumbnail for the news article.'
    )
    meta_url: Optional[MetaUrl] = Field(
        None,
        description='Aggregated information on the URL associated with the news search result.',
    )
    extra_snippets: Optional[List[str]] = Field(
        None,
        description='A list of extra alternate snippets for the news search result.',
    )


class Query(TinyModel):
    """A model representing information gathered around the requested query."""

    original: str = Field(..., description='The original query that was requested.')
    altered: Optional[str] = Field(
        None,
        description='The altered query by the spellchecker. '
        'This is the query that is used to search if any.',
    )
    cleaned: Optional[str] = Field(
        None,
        description='The cleaned normalized query by the spellchecker. '
        'This is the query that is used to search if any.',
    )
    spellcheck_off: Optional[bool] = Field(
        None, description='Whether the spellchecker is enabled or disabled.'
    )
    show_strict_warning: Optional[bool] = Field(
        None,
        description=(
            'True if lack of results is due to strict safesearch setting '
            '(adult content blocked).'
        ),
    )


class NewsSearchApiResponse(TinyModel):
    """Top level response model for successful News Search API requests."""

    type: Literal['news'] = Field('news', description='The type of search API result.')
    query: Query = Field(..., description='News search query string.')
    results: List[NewsResult] = Field(
        ..., description='The list of news results for the given query.'
    )
