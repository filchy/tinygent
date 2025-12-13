from pydantic import Field

from tinygent.types.io.llm_io_input import TinyLLMInput
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.factory import build_llm
from tinygent.tools.tool import tool
from tinygent.types.base import TinyModel


class GetWeatherInput(TinyModel):
    location: str = Field(..., description='The location to get the weather for.')


@tool
def get_weather(data: GetWeatherInput) -> str:
    """Get the current weather in a given location."""

    return f'The weather in {data.location} is sunny with a high of 75°F.'


class GetTimeInput(TinyModel):
    location: str = Field(..., description='The location to get the time for.')


@tool
def get_time(data: GetTimeInput) -> str:
    """Get the current time in a given location."""

    return f'The current time in {data.location} is 2:00 PM.'


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
