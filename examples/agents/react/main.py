from pathlib import Path

from pydantic import Field

from tinygent.agents.react_agent import ActionPromptTemplate
from tinygent.agents.react_agent import PlanPromptTemplate
from tinygent.agents.react_agent import ReactPromptTemplate
from tinygent.agents.react_agent import TinyReActAgent
from tinygent.llms.openai import OpenAILLM
from tinygent.logging import setup_logger
from tinygent.tools.tool import tool
from tinygent.types import TinyModel
from tinygent.utils.load_file import load_yaml

logger = setup_logger('info')


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
    destinations = ['Paris', 'New York', 'Tokyo', 'Barcelona', 'Rome']
    return destinations[: data.top_k]


def main():
    react_agent_prompt = load_yaml(str(Path(__file__).parent / 'react_agent.yaml'))

    react_agent = TinyReActAgent(
        llm=OpenAILLM(),
        prompt_template=ReactPromptTemplate(
            acter=ActionPromptTemplate(
                system=react_agent_prompt['acter']['system'],
                final_answer=react_agent_prompt['acter']['final_answer'],
            ),
            plan=PlanPromptTemplate(
                init_plan=react_agent_prompt['planner']['init_plan'],
                update_plan=react_agent_prompt['planner']['update_plan'],
            ),
        ),
        tools=[get_weather, get_best_destination],
    )

    result = react_agent.run(
        'What is the best travel destination and what is the weather like there?'
    )
    logger.info(f'[RESULT] {result}')


if __name__ == '__main__':
    main()
