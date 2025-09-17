from pydantic import Field
from seznam.services.toolhub import ToolHubService

from tinygent.tools.tool import tool
from tinygent.types import TinyModel


class NakupniExpertReviewsInput(TinyModel):
    product_name: str = Field(
        ..., description='The name of the product to search for expert reviews.'
    )


@tool(use_cache=True)
async def nakupni_expert_reviews(data: NakupniExpertReviewsInput) -> dict:
    """Fetches expert reviews for a given product."""
    return await ToolHubService.call_tool_json(
        'nakupni.expert_reviews', data.model_dump()
    )


class NakupniVideoSearchInput(TinyModel):
    product_name: str = Field(
        ..., description='The name of the product to search for videos.'
    )


@tool(use_cache=True)
async def nakupni_video_search(data: NakupniVideoSearchInput) -> dict:
    """Fetches videos related to a given product."""
    return await ToolHubService.call_tool_json('nakupni.video_search', data.model_dump())


class NakupniCategorySearchInput(TinyModel):
    category: str = Field(..., description='The category to search for products.')
    params_description: str = Field(
        ...,
        description='Description of additional filters or parameters for the search.',
    )


@tool(use_cache=True)
async def nakupni_category_search(data: NakupniCategorySearchInput) -> dict:
    """Fetches products from a specific category with optional filters."""
    return await ToolHubService.call_tool_json(
        'nakupni.category_search', data.model_dump()
    )
