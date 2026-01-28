from pathlib import Path
from typing import Any
import uuid

from pydantic import Field

from tiny_brave import NewsSearchApiResponse
from tiny_brave import NewsSearchRequest
from tiny_brave import brave_news_search
import tiny_chat as tc
from tinygent.agents import TinyReActAgent
from tinygent.agents.middleware import TinyBaseMiddleware
from tinygent.cli.utils import discover_and_register_components
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.factory import build_llm
from tinygent.core.types import TinyModel
from tinygent.logging import setup_logger
from tinygent.memory import BufferChatMemory
from tinygent.prompts import ReActPromptTemplate
from tinygent.tools import register_tool
from tinygent.utils import tiny_yaml_load

logger = setup_logger('debug')

discover_and_register_components()


class BraveNewsConfig(TinyModel):
    query: str = Field(..., description='The search query string.')


@register_tool
async def brave_news(data: BraveNewsConfig):
    result = await brave_news_search(NewsSearchRequest(q=data.query))

    return result


class ChatClientMiddleware(TinyBaseMiddleware):
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

        try:
            news_response = NewsSearchApiResponse.model_validate(result)
            for article in news_response.results:
                await tc.AgentSourceMessage(
                    parent_id=run_id,
                    name=article.title,
                    url=article.url,
                    favicon=article.meta_url.favicon if article.meta_url else None,
                    description=article.description,
                ).send()

        except Exception:
            logger.exception('Failed to parse tool call.')


agent = TinyReActAgent(
    llm=build_llm('openai:gpt-4o'),
    tools=[brave_news],
    memory=BufferChatMemory(),
    prompt_template=ReActPromptTemplate(
        **tiny_yaml_load(
            str(Path(__file__).parent.parent / 'agents' / 'react' / 'prompts.yaml')
        )
    ),
    middleware=[ChatClientMiddleware()],
)


@tc.on_message
async def handle_message(msg: tc.BaseMessage):
    agent_history = tc.current_session.get('agent_history', [])

    async for _ in agent.run_stream(msg.content, history=agent_history):
        pass

    tc.current_session.set('agent_history', agent.memory.copy_chat_messages())


if __name__ == '__main__':
    tc.run(reload=True, log_level='debug', access_log=False)
