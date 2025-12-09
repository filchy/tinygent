from pathlib import Path

from pydantic import Field

from tinygent.agents.multi_step_agent import MultiStepPromptTemplate
from tinygent.agents.multi_step_agent import TinyMultiStepAgent
from tinygent.agents.react_agent import ReActPromptTemplate
from tinygent.agents.react_agent import TinyReActAgent
from tinygent.agents.squad_agent import AgentSquadMember
from tinygent.agents.squad_agent import SquadPromptTemplate
from tinygent.agents.squad_agent import TinySquadAgent
from tinygent.factory import build_llm
from tinygent.logging import setup_logger
from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.memory.buffer_window_chat_memory import BufferWindowChatMemory
from tinygent.tools.reasoning_tool import register_reasoning_tool
from tinygent.tools.tool import register_tool
from tinygent.types.base import TinyModel
from tinygent.utils.yaml import tiny_yaml_load

logger = setup_logger('debug')

# NOTE: Using @register_tool & @register_reasoning_tool decorator to register tools globally,
# allowing them to be discovered and reused by:
# - quick.py via discover_and_register_components()
# - CLI terminal command via config-based agent building


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
    squad_agent = TinySquadAgent(
        llm=build_llm('openai:gpt-4o', temperature=0.1),
        prompt_template=SquadPromptTemplate(
            **tiny_yaml_load(str(Path(__file__).parent / 'prompts.yaml'))
        ),
        squad=[
            AgentSquadMember(
                name='weather_agent',
                description='An agent that provides weather information.',
                agent=TinyReActAgent(
                    llm=build_llm('openai:gpt-4o', temperature=0.1),
                    max_iterations=3,
                    memory=BufferChatMemory(),
                    tools=[get_weather],
                    prompt_template=ReActPromptTemplate(
                        **tiny_yaml_load(
                            str(Path(__file__).parent.parent / 'react' / 'prompts.yaml')
                        )
                    ),
                ),
            ),
            AgentSquadMember(
                name='geoghraphic_agent',
                description='An agent that provides geographic information.',
                agent=TinyMultiStepAgent(
                    llm=build_llm('openai:gpt-4o', temperature=0.1),
                    memory=BufferWindowChatMemory(k=3),
                    tools=[get_best_destination],
                    prompt_template=MultiStepPromptTemplate(
                        **tiny_yaml_load(
                            str(
                                Path(__file__).parent.parent
                                / 'multi-step'
                                / 'prompts.yaml'
                            )
                        )
                    ),
                ),
            ),
        ],
        memory=BufferChatMemory(),
        tools=[calculate_sum],
    )

    result = squad_agent.run(
        'What is the best travel destination and what is the weather like there?'
    )

    logger.info('[RESULT] %s', result)
    logger.info('[MEMORY] %s', squad_agent.memory.load_variables())
    logger.info('[AGENT SUMMARY] %s', str(squad_agent))


if __name__ == '__main__':
    main()
