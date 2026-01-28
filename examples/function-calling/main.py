from pydantic import Field

from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.factory import build_llm
from tinygent.core.types import TinyLLMInput
from tinygent.core.types import TinyModel  # Only needed for Variant 1
from tinygent.tools import tool


# Variant 1: TinyModel descriptor (explicit schema with field descriptions)
class GetWeatherInput(TinyModel):
    location: str = Field(..., description='The location to get the weather for.')


@tool
def get_weather(data: GetWeatherInput) -> str:
    """Get the current weather in a given location."""

    return f'The weather in {data.location} is sunny with a high of 75°F.'


# Variant 2: Regular parameters (auto-generated schema)
@tool
def get_time(location: str) -> str:
    """Get the current time in a given location."""

    return f'The current time in {location} is 2:00 PM.'


if __name__ == '__main__':
    my_tools = [get_weather, get_time]

    openai_llm = build_llm('openai:gpt-4o-mini', temperature=0.1)

    response = openai_llm.generate_with_tools(
        llm_input=TinyLLMInput(
            messages=[TinyHumanMessage(content='What is the weather like in New York?')]
        ),
        tools=my_tools,
    )

    tool_map = {tool.info.name: tool for tool in my_tools}

    for message in response.tiny_iter():
        if message.type == 'chat':
            print(f'LLM response: {message.content}')

        elif message.type == 'tool':
            result = tool_map[message.tool_name](**message.arguments)

            print(
                'Tool %s called with arguments %s, result: %s'
                % (message.tool_name, message.arguments, result)
            )
