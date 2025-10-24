from typing import Any

from tiny_brave.client import TinyBraveClient
from tiny_brave.datamodels.requests.images import ImagesSearchReuest
from tiny_brave.datamodels.requests.news import NewsSearchRequest
from tiny_brave.datamodels.requests.videos import VideoSearchRequest
from tiny_brave.datamodels.requests.web import WebSearchRequest


async def brave_news_search(data: NewsSearchRequest) -> dict[str, Any]:
    """Perform a news search using the Brave Search API."""

    result = await TinyBraveClient().news(data)
    return result.model_dump()


async def brave_web_search(data: WebSearchRequest) -> dict[str, Any]:
    """Perform a web search using the Brave Search API."""

    result = await TinyBraveClient().web(data)
    return result.model_dump()


async def brave_images_search(data: ImagesSearchReuest) -> dict[str, Any]:
    """Perform an image search using the Brave Search API."""

    result = await TinyBraveClient().images(data)
    return result.model_dump()


async def brave_videos_search(data: VideoSearchRequest) -> dict[str, Any]:
    """Perform a video search using the Brave Search API."""

    result = await TinyBraveClient().videos(data)
    return result.model_dump()


if __name__ == '__main__':

    async def main():
        news = await brave_news_search(
            NewsSearchRequest(
                q='Brave Search API',
            )
        )
        print('News Search result: %s', news)

        web = await brave_web_search(
            WebSearchRequest(
                q='Brave Search API',
            )
        )
        print('Web Search result: %s', web)

        images = await brave_images_search(
            ImagesSearchReuest(
                q='Brave Search API',
            )
        )
        print('Image Search result: %s', images)

        videos = await brave_videos_search(
            VideoSearchRequest(
                q='Brave Search API',
            )
        )
        print('Video Search result: %s', videos)

    import asyncio

    asyncio.run(main())
