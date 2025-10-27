from pathlib import Path

from pydantic import Field

from tinygent.agents.multi_step_agent import ActionPromptTemplate
from tinygent.agents.multi_step_agent import FallbackAnswerPromptTemplate
from tinygent.agents.multi_step_agent import MultiStepPromptTemplate
from tinygent.agents.multi_step_agent import PlanPromptTemplate
from tinygent.agents.multi_step_agent import TinyMultiStepAgent
from tinygent.llms.base import init_llm
from tinygent.logging import setup_general_loggers
from tinygent.logging import setup_logger
from tinygent.tools.reasoning_tool import reasoning_tool
from tinygent.tools.tool import tool
from tinygent.types.base import TinyModel
from tinygent.utils.yaml import tiny_yaml_load

logger = setup_logger('debug')
setup_general_loggers('warning')


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
    multi_step_agent_prompt = tiny_yaml_load(str(Path(__file__).parent / 'prompts.yaml'))

    multi_step_agent = TinyMultiStepAgent(
        llm=init_llm('openai:gpt-4o', temperature=0.1),
        prompt_template=MultiStepPromptTemplate(
            acter=ActionPromptTemplate(
                system=multi_step_agent_prompt['acter']['system'],
                final_answer=multi_step_agent_prompt['acter']['final_answer'],
            ),
            plan=PlanPromptTemplate(
                init_plan=multi_step_agent_prompt['plan']['init_plan'],
                update_plan=multi_step_agent_prompt['plan']['update_plan'],
            ),
            fallback=FallbackAnswerPromptTemplate(
                fallback_answer=multi_step_agent_prompt['fallback']['fallback_answer']
            ),
        ),
        tools=[get_weather, get_best_destination],
    )

    result = multi_step_agent.run(
        'What is the best travel destination and what is the weather like there?'
    )

    logger.info(f'[RESULT] {result}')
    logger.info(f'[MEMORY] {multi_step_agent.memory.load_variables()}')
    logger.info(f'[AGENT SUMMARY] {str(multi_step_agent)}')


if __name__ == '__main__':
    main()
