from pathlib import Path
from typing import Any
import uuid

from pydantic import Field

from tiny_brave import NewsSearchApiResponse
from tiny_brave import NewsSearchRequest
from tiny_brave import brave_news_search
import tiny_chat as tc
from tiny_chat.utils import tinychat_2_tinygent_message
from tinygent.agents.react_agent import ReActPromptTemplate
from tinygent.agents.react_agent import TinyReActAgent
from tinygent.cli.utils import discover_and_register_components
from tinygent.datamodels.tool import AbstractTool
from tinygent.llms.base import init_llm
from tinygent.logging import setup_logger
from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.tools.tool import register_tool
from tinygent.types.base import TinyModel
from tinygent.utils.yaml import tiny_yaml_load

logger = setup_logger('debug')

discover_and_register_components()


class BraveNewsConfig(TinyModel):
    query: str = Field(..., description='The search query string.')


@register_tool
async def brave_news(data: BraveNewsConfig):
    result = await brave_news_search(NewsSearchRequest(q=data.query))

    return result


async def answer_hook(*, run_id: str, answer: str):
    await tc.AgentMessage(
        id=run_id,
        content=answer,
    ).send()


async def answer_chunk_hook(*, run_id: str, chunk: str, idx: str):
    await tc.AgentMessageChunk(
        id=run_id,
        content=chunk,
    ).send()


async def tool_call_hook(
    *, run_id: str, tool: AbstractTool, args: dict[str, Any], result: Any
):
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
    llm=init_llm('openai:gpt-4o'),
    tools=[brave_news],
    memory=BufferChatMemory(),
    prompt_template=ReActPromptTemplate(
        **tiny_yaml_load(
            str(Path(__file__).parent.parent / 'agents' / 'react' / 'prompts.yaml')
        )
    ),
)

agent.on_answer = answer_hook
agent.on_answer_chunk = answer_chunk_hook
agent.on_after_tool_call = tool_call_hook


@tc.on_message
async def handle_message(msg: tc.BaseMessage, history: list[tc.BaseMessage]):
    agent_hist = [tinychat_2_tinygent_message(m) for m in history]

    async for _ in agent.run_stream(msg.content, history=agent_hist):
        pass


if __name__ == '__main__':
    tc.run(reload=True, log_level='debug', access_log=False)
