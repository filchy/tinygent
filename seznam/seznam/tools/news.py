from pydantic import Field
from seznam.services.toolhub import ToolHubService

from tinygent.tools.tool import tool
from tinygent.types import TinyModel


class NewsQueryInput(TinyModel):
    query: str = Field(..., description='The search query for news articles.')


@tool(use_cache=True)
async def news_query(data: NewsQueryInput) -> dict:
    """Queries news articles based on a search query."""
    return await ToolHubService.call_tool_json('news.query', data.model_dump())


class OrganicQueryInput(TinyModel):
    query: str = Field(..., description='The search query for organic results.')


@tool(use_cache=True)
async def organic_query(data: OrganicQueryInput) -> dict:
    """Queries organic search results based on a search query."""
    return await ToolHubService.call_tool_json('news.org_query', data.model_dump())


class WeatherQueryInput(TinyModel):
    query: str = Field(..., description='The query for weather information.')


@tool(use_cache=True)
async def weather_query(data: WeatherQueryInput) -> dict:
    """Queries weather information based on a search query."""
    return await ToolHubService.call_tool_json('news.weather', data.model_dump())


class NewsSummarizationInput(TinyModel):
    query: str = Field(..., description='The query for news summarization.')


@tool(use_cache=True)
async def news_summarization(data: NewsSummarizationInput) -> dict:
    """Summarizes news articles based on a search query."""
    return await ToolHubService.call_tool_json('news.summarization', data.model_dump())


class NewsTrendsInput(TinyModel):
    query: str = Field(..., description='The query for news trends.')


@tool(use_cache=True)
async def news_trends(data: NewsTrendsInput) -> dict:
    """Queries news trends based on a search query."""
    return await ToolHubService.call_tool_json('news.latest_news', data.model_dump())


class NewsTrendsCZInput(TinyModel):
    query: str = Field(..., description='The query for Czech news trends.')


@tool(use_cache=True)
async def news_trends_cz(data: NewsTrendsCZInput) -> dict:
    """Queries Czech news trends based on a search query."""
    return await ToolHubService.call_tool_json(
        'news.latest_news_domestic', data.model_dump()
    )


class NewsTrendsWorldInput(TinyModel):
    query: str = Field(..., description='The query for world news trends.')


@tool(use_cache=True)
async def news_trends_world(data: NewsTrendsWorldInput) -> dict:
    """Queries world news trends based on a search query."""
    return await ToolHubService.call_tool_json(
        'news.latest_news_international', data.model_dump()
    )


class NewsTrendsSportInput(TinyModel):
    query: str = Field(..., description='The query for sports news trends.')


@tool(use_cache=True)
async def news_trends_sport(data: NewsTrendsSportInput) -> dict:
    """Queries sports news trends based on a search query."""
    return await ToolHubService.call_tool_json(
        'news.latest_news_sport', data.model_dump()
    )
