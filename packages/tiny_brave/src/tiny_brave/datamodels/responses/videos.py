from typing import List

from pydantic import Field

from tinygent.core.types.base import TinyModel


class Query(TinyModel):
    original: str = Field(..., description='The original query that was requested.')
    altered: str | None = Field(
        default=None, description='The altered query by the spellchecker (if any).'
    )
    cleaned: str | None = Field(
        default=None,
        description='The cleaned normalized query by the spellchecker (if any).',
    )
    spellcheck_off: bool | None = Field(
        default=None, description='Whether the spellchecker is enabled or disabled.'
    )
    show_strict_warning: bool | None = Field(
        default=None,
        description=(
            'True if results are missing due to strict safesearch. '
            'Adult content was blocked by safesearch.'
        ),
    )


class Thumbnail(TinyModel):
    src: str = Field(..., description='The served URL of the thumbnail.')
    original: str | None = Field(
        default=None, description='The original URL of the thumbnail.'
    )


class Profile(TinyModel):
    name: str = Field(..., description='The name of the profile.')
    long_name: str | None = Field(
        default=None, description='The long name of the profile.'
    )
    url: str = Field(..., description='The original URL where the profile is available.')
    img: str | None = Field(
        default=None, description='The served image URL representing the profile.'
    )


class VideoData(TinyModel):
    duration: str | None = Field(
        default=None,
        description='A time string representing the duration of the video.',
    )
    views: int | None = Field(
        default=None, description='The number of views of the video.'
    )
    creator: str | None = Field(default=None, description='The creator of the video.')
    publisher: str | None = Field(
        default=None, description='The publisher of the video.'
    )
    requires_subscription: bool | None = Field(
        default=None, description='Whether the video requires a subscription.'
    )
    tags: List[str] | None = Field(
        default=None, description='A list of tags relevant to the video.'
    )
    author: Profile | None = Field(
        default=None, description='A profile associated with the video.'
    )


class MetaUrl(TinyModel):
    scheme: str | None = Field(
        default=None, description='The protocol scheme extracted from the URL.'
    )
    netloc: str | None = Field(
        default=None, description='The network location part extracted from the URL.'
    )
    hostname: str | None = Field(
        default=None, description='The lowercased domain name extracted from the URL.'
    )
    favicon: str | None = Field(
        default=None, description='The favicon used for the URL.'
    )
    path: str | None = Field(
        default=None,
        description='The hierarchical path of the URL useful as a display string.',
    )


class VideoResult(TinyModel):
    type: str = Field(
        'video_result',
        description='The type of video search API result. Always video_result.',
    )
    url: str = Field(..., description='The source URL of the video.')
    title: str = Field(..., description='The title of the video.')
    description: str | None = Field(
        default=None, description='The description for the video.'
    )
    age: str | None = Field(
        default=None, description='A human readable representation of the page age.'
    )
    page_age: str | None = Field(
        default=None, description='The page age found from the source web page.'
    )
    page_fetched: str | None = Field(
        default=None,
        description='The ISO date time when the page was last fetched (YYYY-MM-DDTHH:MM:SSZ).',
    )
    thumbnail: Thumbnail | None = Field(
        default=None, description='The thumbnail for the video.'
    )
    video: VideoData | None = Field(default=None, description='Metadata for the video.')
    meta_url: MetaUrl | None = Field(
        default=None,
        description='Aggregated information on the URL associated with the video result.',
    )


class Extra(TinyModel):
    might_be_offensive: bool = Field(
        ...,
        description='Indicates whether the video search results might contain offensive content.',
    )


class VideoSearchApiResponse(TinyModel):
    type: str = Field(
        'videos', description='The type of search API result. Always videos.'
    )
    query: Query = Field(..., description='Video search query string.')
    results: List[VideoResult] = Field(
        ..., description='The list of video results for the given query.'
    )
    extra: Extra = Field(
        ..., description='Additional information about the video search results.'
    )
