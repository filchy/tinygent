from pathlib import Path

from pydantic import Field

from tinygent.agents.react_agent import ActionPromptTemplate
from tinygent.agents.react_agent import FallbackPromptTemplate
from tinygent.agents.react_agent import ReActPromptTemplate
from tinygent.agents.react_agent import ReasonPromptTemplate
from tinygent.agents.react_agent import TinyReActAgent
from tinygent.llms.base import init_llm
from tinygent.logging import setup_general_loggers
from tinygent.logging import setup_logger
from tinygent.tools.tool import tool
from tinygent.types.base import TinyModel
from tinygent.utils.yaml import tiny_yaml_load

logger = setup_logger('debug')
setup_general_loggers('warning')


class WeatherInput(TinyModel):
    location: str = Field(..., description='The location to get the weather for.')


@tool
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
    react_agent_prompt = tiny_yaml_load(str(Path(__file__).parent / 'prompts.yaml'))

    react_agent = TinyReActAgent(
        llm=init_llm('openai:gpt-4o', temperature=0.1),
        max_iterations=3,
        prompt_template=ReActPromptTemplate(
            reason=ReasonPromptTemplate(
                init=react_agent_prompt['reason']['init'],
                update=react_agent_prompt['reason']['update'],
            ),
            action=ActionPromptTemplate(action=react_agent_prompt['action']['action']),
            fallback=FallbackPromptTemplate(
                fallback_answer=react_agent_prompt['fallback']['fallback_answer']
            ),
        ),
        tools=[get_weather, get_best_destination],
    )

    result = react_agent.run(
        'What is the best travel destination and what is the weather like there?'
    )

    logger.info(f'[RESULT] {result}')
    logger.info(f'[MEMORY] {react_agent.memory.load_variables()}')
    logger.info(f'[AGENT SUMMARY] {str(react_agent)}')


if __name__ == '__main__':
    main()
