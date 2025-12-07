from pathlib import Path

from pydantic import Field

from tinygent.agents.react_agent import ReActPromptTemplate
from tinygent.agents.react_agent import TinyReActAgent
from tinygent.llms.base import init_llm
from tinygent.logging import setup_logger
from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.tools.tool import register_tool
from tinygent.types.base import TinyModel
from tinygent.utils.yaml import tiny_yaml_load

logger = setup_logger('debug')

# NOTE: Using @register_tool decorator to register tools globally,
# allowing them to be discovered and reused by:
# - quick.py via discover_and_register_components()
# - CLI terminal command via config-based agent building


class WeatherInput(TinyModel):
    location: str = Field(..., description='The location to get the weather for.')


@register_tool
def get_weather(data: WeatherInput) -> str:
    """Get the current weather in a given location."""

    return f'The weather in {data.location} is sunny with a high of 75°F.'


class GetBestDestinationInput(TinyModel):
    top_k: int = Field(..., description='The number of top destinations to return.')


@register_tool
def get_best_destination(data: GetBestDestinationInput) -> list[str]:
    """Get the best travel destinations."""
    destinations = {'Paris', 'New York', 'Tokyo', 'Barcelona', 'Rome'}
    return list(destinations)[: data.top_k]


async def main():
    react_agent_prompt = tiny_yaml_load(str(Path(__file__).parent / 'prompts.yaml'))

    react_agent = TinyReActAgent(
        llm=init_llm('openai:gpt-4o', temperature=0.1),
        max_iterations=3,
        memory=BufferChatMemory(),
        prompt_template=ReActPromptTemplate(**react_agent_prompt),
        tools=[get_weather, get_best_destination],
    )

    result: str = ''
    async for chunk in react_agent.run_stream(
        'What is the best travel destination and what is the weather like there?'
    ):
        logger.info(f'[STREAM CHUNK] {chunk}')
        result += chunk

    logger.info(f'[RESULT] {result}')
    logger.info(f'[MEMORY] {react_agent.memory.load_variables()}')
    logger.info(f'[AGENT SUMMARY] {str(react_agent)}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
