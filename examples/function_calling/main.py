from langchain_core.prompt_values import StringPromptValue
from pydantic import BaseModel
from pydantic import Field

from tinygent.llms.openai import OpenAILLM
from tinygent.tools.tool import tool


class GetWeatherInput(BaseModel):

    location: str = Field(..., description='The location to get the weather for.')


@tool
def get_weather(data: GetWeatherInput) -> str:
    """Get the current weather in a given location."""

    return f'The weather in {data.location} is sunny with a high of 75°F.'


class GetTimeInput(BaseModel):

    location: str = Field(..., description='The location to get the time for.')


@tool
def get_time(data: GetTimeInput) -> str:
    """Get the current time in a given location."""

    return f'The current time in {data.location} is 2:00 PM.'


if __name__ == '__main__':
    my_tools = [get_weather, get_time]
    openai_llm = OpenAILLM()

    response = openai_llm.generate_with_tools(
        prompt=StringPromptValue(
            text='What is the weather like in New York?'
        ),
        tools=my_tools
    )

    for message in response.tiny_iter():
        print(f'{message.type}: {message}')
