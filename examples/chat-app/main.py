from typing import Any
import uuid

from pydantic import Field

from tiny_brave import NewsSearchApiResponse
from tiny_brave import NewsSearchRequest
from tiny_brave import WebSearchApiResponse
from tiny_brave import WebSearchRequest
from tiny_brave import brave_news_search
from tiny_brave import brave_web_search
import tiny_chat as tc
from tinygent.agents.middleware import TinyBaseMiddleware
from tinygent.agents.middleware import TinyToolCallLimiterMiddleware
from tinygent.cli.utils import discover_and_register_components
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.factory import build_agent
from tinygent.core.types import TinyModel
from tinygent.logging import setup_logger
from tinygent.memory import BufferChatMemory
from tinygent.tools import register_tool

logger = setup_logger('debug')

discover_and_register_components()


class BraveConfig(TinyModel):
    query: str = Field(..., description='The search query string.')


@register_tool
async def brave_news(data: BraveConfig):
    """Search recent news articles from news publishers (headlines, journalism, press, media coverage)."""
    raw = await brave_news_search(NewsSearchRequest(q=data.query))

    return NewsSearchApiResponse.model_validate(raw)


@register_tool
async def brave_web(data: BraveConfig):
    """Search general web pages (blogs, docs, forums, websites, informational pages)."""
    raw = await brave_web_search(WebSearchRequest(q=data.query))

    return WebSearchApiResponse.model_validate(raw)


class ChatClientMiddleware(TinyBaseMiddleware):
    @staticmethod
    async def _send_sources(run_id: str, result: Any) -> bool:
        if not isinstance(result, (NewsSearchApiResponse, WebSearchApiResponse)):
            return False

        items = (
            result.results
            if isinstance(result, NewsSearchApiResponse)
            else result.web.results
            if result.web
            else []
        )

        for item in items:
            await tc.AgentSourceMessage(
                parent_id=run_id,
                name=item.title,
                url=item.url,
                favicon=item.meta_url.favicon if item.meta_url else None,
                description=item.description,
            ).send()

        return True

    async def on_answer(
        self, *, run_id: str, answer: str, kwargs: dict[str, Any]
    ) -> None:
        await tc.AgentMessage(
            id=run_id,
            content=answer,
        ).send()

    async def on_answer_chunk(
        self, *, run_id: str, chunk: str, idx: str, kwargs: dict[str, Any]
    ) -> None:
        await tc.AgentMessageChunk(
            id=run_id,
            content=chunk,
        ).send()

    async def after_tool_call(
        self,
        *,
        run_id: str,
        tool: AbstractTool,
        args: dict[str, Any],
        result: Any,
        kwargs: dict[str, Any],
    ) -> None:
        await tc.AgentToolCallMessage(
            id=str(uuid.uuid4()),
            parent_id=run_id,
            tool_name=tool.info.name,
            tool_args=args,
            content=result,
        ).send()

        if not await self._send_sources(run_id, result):
            logger.exception('Failed to parse tool call.')


agent = build_agent(
    'react',
    llm='openai:gpt-4o',
    tools=[brave_news, brave_web],
    memory=BufferChatMemory(),
    middleware=[
        ChatClientMiddleware(),
        TinyToolCallLimiterMiddleware(tool_name='brave_web', max_tool_calls=1),
    ],
)


@tc.on_message
async def handle_message(msg: tc.BaseMessage):
    agent_history = tc.current_session.get('agent_history', [])

    async for _ in agent.run_stream(msg.content, history=agent_history):
        pass

    tc.current_session.set('agent_history', agent.memory.copy_chat_messages())


if __name__ == '__main__':
    tc.run(reload=True, log_level='debug', access_log=False)
