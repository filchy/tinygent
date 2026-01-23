import logging
import os
from typing import TypeVar
from urllib.parse import urljoin

import httpx

from tiny_brave.constants import BASE_URL
from tiny_brave.constants import DEFAULT_MAX_RETRIES
from tiny_brave.constants import DEFAULT_TIMEOUT
from tiny_brave.datamodels.endpoints import BraveEndpoint
from tiny_brave.datamodels.requests.images import ImagesSearchReuest
from tiny_brave.datamodels.requests.news import NewsSearchRequest
from tiny_brave.datamodels.requests.videos import VideoSearchRequest
from tiny_brave.datamodels.requests.web import WebSearchRequest
from tiny_brave.datamodels.responses.images import ImageSearchApiResponse
from tiny_brave.datamodels.responses.news import NewsSearchApiResponse
from tiny_brave.datamodels.responses.videos import VideoSearchApiResponse
from tiny_brave.datamodels.responses.web import WebSearchApiResponse
from tiny_brave.exceptions import TinyBraveAPIError
from tiny_brave.exceptions import TinyBraveClientError
from tinygent.core.types.base import TinyModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=TinyModel)

RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.WriteError,
)


class TinyBraveClient:
    def __init__(self):
        if not (brave_token := os.getenv('BRAVE_API_KEY')):
            raise TinyBraveClientError("'BRAVE_API_KEY' environment variable not set.")

        self._base_url = BASE_URL

        self._headers = {
            'X-Subscription-Token': brave_token,
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'Cache-Control': 'no-cache',
        }

    async def _get(
        self,
        endpoint: BraveEndpoint,
        params: dict[str, str] | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> httpx.Response:
        last_exc: Exception | None = None
        url = urljoin(self._base_url, f'{endpoint.value}/search')

        async with httpx.AsyncClient(
            headers=self._headers,
            timeout=timeout,
        ) as client:
            for attempt in range(1, max_retries + 1):
                try:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    return response

                except RETRYABLE_EXCEPTIONS as e:
                    last_exc = e
                    logger.warning(
                        'Retryable network error calling %s: %s (attempt %d/%d)',
                        url,
                        e,
                        attempt,
                        max_retries,
                    )

                except httpx.HTTPStatusError as e:
                    logger.error(
                        'HTTP error calling %s: %s (status %d) - not retrying',
                        url,
                        e,
                        e.response.status_code,
                    )
                    raise TinyBraveClientError(
                        f'HTTP error {e.response.status_code} calling {url}: {e}'
                    )

            raise TinyBraveAPIError(
                f'Failed to fetch data from {url} after {max_retries} attempts.'
            ) from last_exc

    async def _use_brave(
        self,
        endpoint: BraveEndpoint,
        request: TinyModel,
        response_model: type[T],
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> T:
        response = await self._get(
            endpoint,
            params=request.model_dump(exclude_none=True, by_alias=True),
            max_retries=max_retries,
            timeout=timeout,
        )
        return response_model.model_validate(response.json())

    async def news(
        self,
        request: NewsSearchRequest,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> NewsSearchApiResponse:
        return await self._use_brave(
            BraveEndpoint.news,
            request=request,
            response_model=NewsSearchApiResponse,
            max_retries=max_retries,
            timeout=timeout,
        )

    async def web(
        self,
        request: WebSearchRequest,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> WebSearchApiResponse:
        return await self._use_brave(
            BraveEndpoint.web,
            request=request,
            response_model=WebSearchApiResponse,
            max_retries=max_retries,
            timeout=timeout,
        )

    async def images(
        self,
        request: ImagesSearchReuest,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> ImageSearchApiResponse:
        return await self._use_brave(
            BraveEndpoint.images,
            request=request,
            response_model=ImageSearchApiResponse,
            max_retries=max_retries,
            timeout=timeout,
        )

    async def videos(
        self,
        request: VideoSearchRequest,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> VideoSearchApiResponse:
        return await self._use_brave(
            BraveEndpoint.videos,
            request=request,
            response_model=VideoSearchApiResponse,
            max_retries=max_retries,
            timeout=timeout,
        )
