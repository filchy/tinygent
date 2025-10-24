from pydantic import Field

from tinygent.datamodels.llm_io_input import TinyLLMInput
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.llms import OpenAILLM
from tinygent.runtime.global_registry import GlobalRegistry
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

    openai_llm = OpenAILLM()

    response = openai_llm.generate_with_tools(
        llm_input=TinyLLMInput(
            messages=[TinyHumanMessage(content='What is the weather like in New York?')]
        ),
        tools=my_tools,
    )

    for message in response.tiny_iter():
        if message.type == 'chat':
            print(f'LLM response: {message.content}')

        elif message.type == 'tool':
            selected_tool = GlobalRegistry.get_registry().get_tool(message.tool_name)

            result = selected_tool(**message.arguments)

            print(
                'Tool %s called with arguments %s, result: %s'
                % (message.tool_name, message.arguments, result)
            )
