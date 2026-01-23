from pathlib import Path
from typing import Any
import uuid

from pydantic import Field

from tiny_brave import NewsSearchApiResponse
from tiny_brave import NewsSearchRequest
from tiny_brave import brave_news_search
import tiny_chat as tc
from tinygent.agents.middleware.base import AgentMiddleware
from tinygent.agents.react_agent import ReActPromptTemplate
from tinygent.agents.react_agent import TinyReActAgent
from tinygent.cli.utils import discover_and_register_components
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.factory import build_llm
from tinygent.core.types.base import TinyModel
from tinygent.logging import setup_logger
from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.tools.tool import register_tool
from tinygent.utils.yaml import tiny_yaml_load

logger = setup_logger('debug')

discover_and_register_components()


class BraveNewsConfig(TinyModel):
    query: str = Field(..., description='The search query string.')


@register_tool
async def brave_news(data: BraveNewsConfig):
    result = await brave_news_search(NewsSearchRequest(q=data.query))

    return result


class ChatClientMiddleware(AgentMiddleware):
    async def on_answer(self, *, run_id: str, answer: str) -> None:
        await tc.AgentMessage(
            id=run_id,
            content=answer,
        ).send()

    async def on_answer_chunk(self, *, run_id: str, chunk: str, idx: str) -> None:
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
