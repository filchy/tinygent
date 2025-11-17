from pathlib import Path

from pydantic import Field

from tinygent.agents.multi_step_agent import MultiStepPromptTemplate
from tinygent.agents.multi_step_agent import TinyMultiStepAgent
from tinygent.agents.react_agent import ReActPromptTemplate
from tinygent.agents.react_agent import TinyReActAgent
from tinygent.agents.squad_agent import AgentSquadMember
from tinygent.agents.squad_agent import SquadPromptTemplate
from tinygent.agents.squad_agent import TinySquadAgent
from tinygent.llms.base import init_llm
from tinygent.logging import setup_logger
from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.memory.buffer_window_chat_memory import BufferWindowChatMemory
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
    squad_agent = TinySquadAgent(
        llm=init_llm('openai:gpt-4o', temperature=0.1),
        prompt_template=SquadPromptTemplate(
            **tiny_yaml_load(str(Path(__file__).parent / 'prompts.yaml'))
        ),
        squad=[
            AgentSquadMember(
                name='weather_agent',
                description='An agent that provides weather information.',
                agent=TinyReActAgent(
                    llm=init_llm('openai:gpt-4o', temperature=0.1),
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
                    llm=init_llm('openai:gpt-4o', temperature=0.1),
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
    )

    result = squad_agent.run(
        'What is the best travel destination and what is the weather like there?'
    )

    logger.info(f'[RESULT] {result}')
    logger.info(f'[MEMORY] {squad_agent.memory.load_variables()}')
    logger.info(f'[AGENT SUMMARY] {str(squad_agent)}')


if __name__ == '__main__':
    main()
