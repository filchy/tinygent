from tinygent.core.factory import build_agent
from tinygent.logging import setup_logger
from tinygent.tools import tool

setup_logger('debug')


@tool
def get_weather(location: str) -> str:
    """Get the current weather in a given location."""
    return f'The weather in {location} is sunny with a high of 75°F.'


agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[get_weather],
)

print(agent.run('What is the weather like in Prague?'))
