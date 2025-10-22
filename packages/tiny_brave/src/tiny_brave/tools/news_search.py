import logging
from typing import Any

from tiny_brave.client import TinyBraveClient
from tiny_brave.requests.news import NewsSearchRequest

logger = logging.getLogger(__name__)


async def brave_news_search(data: NewsSearchRequest) -> dict[str, Any]:
    """Perform a news search using the Brave Search API."""

    result = await TinyBraveClient().news(data)
    return result.model_dump()


if __name__ == '__main__':
    async def main():
        result = await brave_news_search(
            NewsSearchRequest(
                query='Brave Search API',
            )
        )

        print('Brave Search result: %s', result)

    import asyncio
    asyncio.run(main())
