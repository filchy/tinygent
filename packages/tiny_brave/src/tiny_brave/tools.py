from typing import Any

from tiny_brave.client import TinyBraveClient
from tiny_brave.requests.news import NewsSearchRequest
from tiny_brave.requests.web import WebSearchRequest


async def brave_news_search(data: NewsSearchRequest) -> dict[str, Any]:
    """Perform a news search using the Brave Search API."""

    result = await TinyBraveClient().news(data)
    return result.model_dump()


async def brave_web_search(data: WebSearchRequest) -> dict[str, Any]:
    """Perform a web search using the Brave Search API."""

    result = await TinyBraveClient().web(data)
    return result.model_dump()


if __name__ == '__main__':
    async def main():
        news = await brave_news_search(
            NewsSearchRequest(
                query='Brave Search API',
            )
        )
        print('News Search result: %s', news)

        web = await brave_web_search(
            WebSearchRequest(
                query='Brave Search API',
            )
        )
        print('Web Search result: %s', web)

    import asyncio
    asyncio.run(main())
