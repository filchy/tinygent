from pathlib import Path
from pydantic import Field

from tinygent.agents.squad_agent import TinySquadAgent
from tinygent.llms.base import init_llm
from tinygent.logging import setup_logger
from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.tools.reasoning_tool import reasoning_tool
from tinygent.tools.tool import tool
from tinygent.types.base import TinyModel
from tinygent.utils.yaml import tiny_yaml_load


logger = setup_logger('debug')


class WeatherInput(TinyModel):
    location: str = Field(..., description='The location to get the weather for.')


@reasoning_tool(
    reasoning_prompt='Provide reasoning for why the weather information is needed.'
)
def get_weather(data: WeatherInput) -> str:
    """Get the current weather in a given location."""

    return f'The weather in {data.location} is sunny with a high of 75°F.'


class GetBestDestinationInput(TinyModel):
    top_k: int = Field(..., description='The number of top destinations to return.')


@tool
def get_best_destination(data: GetBestDestinationInput) -> list[str]:
    """Get the best travel destinations."""
    destinations = {'Paris', 'New York', 'Tokyo', 'Barcelona', 'Rome'}
    return list(destinations)[: data.top_k]


def main():
    squad_agent_prompt = tiny_yaml_load(str(Path(__file__).parent / 'prompts.yaml'))

    squad_agent = TinySquadAgent(
        llm=init_llm('openai:gpt-4o', temperature=0.1),
        prompt_template=squad_agent_prompt,
        squad=[],
        memory=BufferChatMemory(),
    )


if __name__ == '__main__':
    main()
