from typing import Any
from typing import Literal

from pydantic import Field

from tinygent.core.types.base import TinyModel


class Thumbnail(TinyModel):
    src: str = Field(
        ...,
        description='The served URL of the picture thumbnail.',
    )
    original: str | None = Field(
        default=None,
        description='The original image URL.',
    )


class MetaUrl(TinyModel):
    scheme: str = Field(
        ...,
        description='The protocol scheme extracted from the URL.',
    )
    netloc: str = Field(
        ...,
        description='The network location part extracted from the URL.',
    )
    hostname: str | None = Field(
        default=None,
        description='The lowercased domain name extracted from the URL.',
    )
    favicon: str = Field(
        ...,
        description='The favicon URL associated with the domain.',
    )
    path: str = Field(
        ...,
        description='The hierarchical path of the URL.',
    )


class Query(TinyModel):
    original: str = Field(
        ...,
        description='The original query that was requested.',
    )
    altered: str | None = Field(
        default=None,
        description='The altered query for which the search was performed.',
    )
    safesearch: bool | None = Field(
        default=None,
        description='Whether safesearch was enabled.',
    )
    is_trending: bool | None = Field(
        default=None,
        description='Whether the query is trending.',
    )
    is_news_breaking: bool | None = Field(
        default=None,
        description='Whether the query has breaking news articles relevant to it.',
    )
    language: dict[str, Any] | None = Field(
        default=None,
        description='Language information gathered from the query.',
    )
    country: str | None = Field(
        default=None,
        description='The country that was used for the query.',
    )


class DiscussionResult(TinyModel):
    type: str = Field(
        default='discussion',
        description='The discussion result type identifier. Always "discussion".',
    )
    data: dict[str, Any] | None = Field(
        default=None,
        description='Enriched aggregated data for the relevant forum post.',
    )


class Discussions(TinyModel):
    type: str = Field(
        default='search',
        description='The type identifying a discussion cluster. Always "search".',
    )
    results: list[DiscussionResult] = Field(
        default_factory=list,
        description='A list of discussion results relevant to the query.',
    )
    mutated_by_goggles: bool | None = Field(
        default=None,
        description='Whether the discussion results were modified by a Goggle.',
    )


class QA(TinyModel):
    question: str = Field(
        ...,
        description='The question being asked.',
    )
    answer: str = Field(
        ...,
        description='The answer to the question.',
    )
    title: str = Field(
        ...,
        description='The title of the post containing the Q&A.',
    )
    url: str = Field(
        ...,
        description='The URL pointing to the post.',
    )
    meta_url: MetaUrl | None = Field(
        default=None,
        description='Aggregated information about the URL.',
    )


class FAQ(TinyModel):
    type: str = Field(
        default='faq',
        description='The FAQ result type identifier. Always "faq".',
    )
    results: list[QA] = Field(
        default_factory=list,
        description='A list of aggregated question-answer results relevant to the query.',
    )


class WebResult(TinyModel):
    type: str = Field(
        default='search_result',
        description='The type identifying a web search result. Always "search_result".',
    )
    title: str = Field(
        ...,
        description='The title of the web page.',
    )
    url: str = Field(
        ...,
        description='The URL of the web page.',
    )
    description: str | None = Field(
        default=None,
        description='A description or snippet of the web page.',
    )
    language: str | None = Field(
        default=None,
        description='The main language on the web page.',
    )
    family_friendly: bool | None = Field(
        default=None,
        description='Whether the web page is family friendly.',
    )
    meta_url: MetaUrl | None = Field(
        default=None,
        description='Aggregated metadata about the URL.',
    )
    thumbnail: Thumbnail | None = Field(
        default=None,
        description='Thumbnail image for the result.',
    )
    age: str | None = Field(
        default=None,
        description='A string representing the age of the web page.',
    )
    extra_snippets: list[str] | None = Field(
        default=None,
        description='A list of extra alternate snippets for the result.',
    )


class WebResults(TinyModel):
    type: str = Field(
        default='search',
        description='The type identifying web search results. Always "search".',
    )
    results: list[WebResult] = Field(
        default_factory=list,
        description='A list of web search results relevant to the query.',
    )
    family_friendly: bool | None = Field(
        default=None,
        description='Whether the search results are family friendly.',
    )


class NewsResult(TinyModel):
    meta_url: MetaUrl | None = Field(
        default=None,
        description='Aggregated metadata about the news article URL.',
    )
    source: str | None = Field(
        default=None,
        description='The source of the news.',
    )
    breaking: bool = Field(
        ...,
        description='Whether the news result is breaking news.',
    )
    is_live: bool = Field(
        ...,
        description='Whether the news result is live.',
    )
    thumbnail: Thumbnail | None = Field(
        default=None,
        description='Thumbnail image for the news result.',
    )
    age: str | None = Field(
        default=None,
        description='A string representing the age of the news article.',
    )
    extra_snippets: list[str] | None = Field(
        default=None,
        description='Extra alternate snippets for the news article.',
    )


class NewsResults(TinyModel):
    type: str = Field(
        default='news',
        description='The type identifying news results. Always "news".',
    )
    results: list[NewsResult] = Field(
        default_factory=list,
        description='A list of news results relevant to the query.',
    )
    mutated_by_goggles: bool | None = Field(
        default=None,
        description='Whether the news results were modified by a Goggle.',
    )


class VideoResult(TinyModel):
    type: str = Field(
        default='video_result',
        description='The type identifying a video result. Always "video_result".',
    )
    video: dict[str, Any] = Field(
        ...,
        description='Metadata for the video.',
    )
    meta_url: MetaUrl | None = Field(
        default=None,
        description='Aggregated metadata about the video URL.',
    )
    thumbnail: Thumbnail | None = Field(
        default=None,
        description='Thumbnail image for the video result.',
    )
    age: str | None = Field(
        default=None,
        description='A string representing the age of the video.',
    )


class VideoResults(TinyModel):
    type: str = Field(
        default='videos',
        description='The type identifying video results. Always "videos".',
    )
    results: list[VideoResult] = Field(
        default_factory=list,
        description='A list of video results relevant to the query.',
    )
    mutated_by_goggles: bool | None = Field(
        default=None,
        description='Whether the video results were modified by a Goggle.',
    )


class WebSearchApiResponse(TinyModel):
    type: Literal['search'] = Field(
        'search',
        description='The type of web search API result',
    )
    query: Query | None = Field(
        default=None,
        description='Search query and its modifications.',
    )
    web: WebResults | None = Field(
        default=None,
        description='Web search results relevant to the query.',
    )
    news: NewsResults | None = Field(
        default=None,
        description='News results relevant to the query.',
    )
    videos: VideoResults | None = Field(
        default=None,
        description='Videos relevant to the query.',
    )
    discussions: Discussions | None = Field(
        default=None,
        description='Discussions aggregated from forum posts relevant to the query.',
    )
    faq: FAQ | None = Field(
        default=None,
        description='Frequently asked questions relevant to the query.',
    )
    infobox: dict[str, Any] | None = Field(
        default=None,
        description='Aggregated information on an entity shown as an infobox.',
    )
    locations: dict[str, Any] | None = Field(
        default=None,
        description='Places of interest relevant to location-sensitive queries.',
    )
    summarizer: dict[str, Any] | None = Field(
        default=None,
        description='Summary key to get summary results for the query.',
    )
    rich: dict[str, Any] | None = Field(
        default=None,
        description='Callback information for rich results.',
    )
    mixed: dict[str, Any] | None = Field(
        default=None,
        description='Preferred ranked order of search results.',
    )
