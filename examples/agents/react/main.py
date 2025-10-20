from pathlib import Path

from pydantic import Field

from tinygent.agents.react_agent import ActionPromptTemplate
from tinygent.agents.react_agent import ReActPromptTemplate
from tinygent.agents.react_agent import ReasonPromptTemplate
from tinygent.agents.react_agent import TinyReActAgent
from tinygent.llms.openai import OpenAILLM
from tinygent.logging import setup_general_loggers
from tinygent.logging import setup_logger
from tinygent.tools.tool import tool
from tinygent.types.base import TinyModel
from tinygent.utils.yaml import tiny_yaml_load

logger = setup_logger('debug')
setup_general_loggers('warning')


class SearchAppleInput(TinyModel):
    query: str = Field(
        ..., description='The search query to look up information about apples.'
    )


@tool
def search_apple_information(data: SearchAppleInput) -> str:
    return 'Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.'


class SearchAmericanHistoryInput(TinyModel):
    query: str = Field(
        ...,
        description='The search query to look up information about American history.',
    )


@tool
def search_american_history(data: SearchAmericanHistoryInput) -> str:
    return 'Gerald Ford'


def main():
    react_agent_prompt = tiny_yaml_load(str(Path(__file__).parent / 'prompts.yaml'))

    react_agent = TinyReActAgent(
        llm=OpenAILLM(),
        prompt_template=ReActPromptTemplate(
            reason=ReasonPromptTemplate(
                init=react_agent_prompt['reason']['init'],
                update=react_agent_prompt['reason']['update'],
            ),
            action=ActionPromptTemplate(action=react_agent_prompt['action']['action']),
        ),
        tools=[search_apple_information, search_american_history],
    )

    result = react_agent.run('Who was the president of the USA when Apple was founded?')

    logger.info(f'Final Result: {result}')


if __name__ == '__main__':
    main()
