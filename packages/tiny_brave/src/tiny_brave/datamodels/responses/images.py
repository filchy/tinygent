from typing import Literal
from typing import Optional

from pydantic import Field

from tinygent.core.types.base import TinyModel


class Query(TinyModel):
    original: str = Field(..., description='The original query that was requested.')
    altered: Optional[str] = Field(
        None, description='The altered query by the spellchecker.'
    )
    spellcheck_off: Optional[bool] = Field(
        None, description='Whether the spellchecker is enabled or disabled.'
    )
    show_strict_warning: Optional[bool] = Field(
        None, description='True if results blocked by strict safesearch setting.'
    )


class Thumbnail(TinyModel):
    src: Optional[str] = Field(None, description='The served URL of the image.')
    width: Optional[int] = Field(None, description='The width of the image.')
    height: Optional[int] = Field(None, description='The height of the image.')


class Properties(TinyModel):
    url: Optional[str] = Field(None, description='The image URL.')
    placeholder: Optional[str] = Field(None, description='The placeholder image URL.')
    width: Optional[int] = Field(None, description='The width of the image.')
    height: Optional[int] = Field(None, description='The height of the image.')


class MetaUrl(TinyModel):
    scheme: Optional[str] = Field(None, description='The protocol scheme from the URL.')
    netloc: Optional[str] = Field(
        None, description='The network location part from the URL.'
    )
    hostname: Optional[str] = Field(
        None, description='The lowercased domain name from the URL.'
    )
    favicon: Optional[str] = Field(None, description='The favicon used for the URL.')
    path: Optional[str] = Field(None, description='The hierarchical path of the URL.')


class ImageResult(TinyModel):
    type: Literal['image_result'] = Field(
        'image_result', description='The type of image search API result.'
    )
    title: Optional[str] = Field(None, description='The title of the image.')
    url: Optional[str] = Field(
        None, description='The original page URL where the image was found.'
    )
    source: Optional[str] = Field(
        None, description='The source domain where the image was found.'
    )
    page_fetched: Optional[str] = Field(
        None, description='ISO datetime when the page was last fetched.'
    )
    thumbnail: Optional[Thumbnail] = Field(
        None, description='The thumbnail for the image.'
    )
    properties: Optional[Properties] = Field(None, description='Metadata for the image.')
    meta_url: Optional[MetaUrl] = Field(
        None, description='Aggregated information on the image URL.'
    )
    confidence: Optional[Literal['low', 'medium', 'high']] = Field(
        None, description='The confidence level for the image result.'
    )


class Extra(TinyModel):
    might_be_offensive: bool = Field(
        ..., description='Indicates if results might be offensive.'
    )


class ImageSearchApiResponse(TinyModel):
    type: Literal['images'] = Field(
        'images', description='The type of search API result.'
    )
    query: Query = Field(..., description='Image search query string.')
    results: list[ImageResult] = Field(
        ..., description='List of image results for the query.'
    )
    extra: Extra = Field(..., description='Additional information about the results.')
