from pathlib import Path

from pydantic import Field

from tinygent.cli.builder import build_agent
from tinygent.cli.utils import discover_and_register_components
from tinygent.logging import setup_logger
from tinygent.tools.reasoning_tool import register_reasoning_tool
from tinygent.tools.tool import register_tool
from tinygent.types.base import TinyModel
from tinygent.utils.yaml import tiny_yaml_load

logger = setup_logger('debug')


class WeatherInput(TinyModel):
    location: str = Field(..., description='The location to get the weather for.')


@register_reasoning_tool(
    reasoning_prompt='Provide reasoning for why the weather information is needed.'
)
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


class SumInput(TinyModel):
    numbers: list[int] = Field(..., description='A list of numbers to sum.')


@register_tool
def calculate_sum(data: SumInput) -> int:
    """Calculate the sum of a list of numbers."""
    return sum(data.numbers)


def main():
    discover_and_register_components()

    agent = build_agent(tiny_yaml_load(str(Path(__file__).parent / 'agent.yaml')))

    result = agent.run('What is the weather like in Paris?')

    logger.info(agent)
    logger.info(f'[RESULT] {result}')


if __name__ == '__main__':
    main()
