import logging
import os
import httpx
from urllib.parse import urljoin

from tinygent.types.base import TinyModel

from tiny_brave.constants import BASE_URL
from tiny_brave.constants import DEFAULT_MAX_RETRIES
from tiny_brave.constants import DEFAULT_TIMEOUT
from tiny_brave.exceptions import TinyBraveAPIError
from tiny_brave.exceptions import TinyBraveClientError
from tiny_brave.requests.news import NewsSearchRequest
from tiny_brave.responses.news import NewsSearchApiResponse
from tiny_brave.types.endpoints import BraveEndpoint

logger = logging.getLogger(__name__)


class TinyBraveClient:
    def __init__(self):
        if not (brave_token := os.getenv('BRAVE_API_KEY')):
            raise TinyBraveClientError(
                '\'BRAVE_API_KEY\' environment variable not set.'
            )

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
        url = urljoin(self._base_url, f'{endpoint.value}/search')

        async with httpx.AsyncClient(
            headers=self._headers,
            timeout=timeout,
        ) as client:
            for _ in range(max_retries):
                try:
                    response = await client.get(
                        url,
                        params=params
                    )
                    response.raise_for_status()
                    return response
                except httpx.HTTPError as e:
                    logger.warning(
                        'Request to %s failed: %s',
                        url,
                        str(e)
                    )
            raise TinyBraveAPIError(
                f'Failed to fetch data from {url} after {max_retries} attempts.'
            )

    async def _use_brave(
        self,
        endpoint: BraveEndpoint,
        request: TinyModel,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> httpx.Response:

        return await self._get(
            endpoint,
            params=request.model_dump(exclude_none=True, by_alias=True),
            max_retries=max_retries,
            timeout=timeout,
        )

    async def news(
        self,
        request: NewsSearchRequest,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> NewsSearchApiResponse:
        response = await self._use_brave(
            BraveEndpoint.news,
            request=request,
            max_retries=max_retries,
            timeout=timeout,
        )

        return NewsSearchApiResponse.model_validate(response.json())
