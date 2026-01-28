from pydantic import Field

from tinygent.agents.middleware import TinyLLMToolSelectorMiddlewareConfig
from tinygent.core.factory import build_agent
from tinygent.core.factory import build_llm
from tinygent.core.factory import build_middleware
from tinygent.core.types import TinyModel
from tinygent.logging import setup_logger
from tinygent.tools import reasoning_tool

logging = setup_logger('debug')


class GreetInput(TinyModel):
    name: str = Field(..., description='The name of the person to greet.')


class CalculateInput(TinyModel):
    a: int = Field(..., description='First number')
    b: int = Field(..., description='Second number')


class WeatherInput(TinyModel):
    location: str = Field(..., description='Location to get weather for')


class NewsInput(TinyModel):
    topic: str = Field(..., description='News topic to search for')


@reasoning_tool
def greet(data: GreetInput) -> str:
    """Return a simple greeting."""
    return f'Hello, {data.name}!'


@reasoning_tool
def add_numbers(data: CalculateInput) -> str:
    """Add two numbers together."""
    result = data.a + data.b
    return f'The sum of {data.a} and {data.b} is {result}'


@reasoning_tool
def multiply_numbers(data: CalculateInput) -> str:
    """Multiply two numbers together."""
    result = data.a * data.b
    return f'The product of {data.a} and {data.b} is {result}'


@reasoning_tool
def divide_numbers(data: CalculateInput) -> str:
    """Divide two numbers."""
    if data.b == 0:
        return 'Error: Division by zero'
    result = data.a / data.b
    return f'The quotient of {data.a} and {data.b} is {result}'


@reasoning_tool
def subtract_numbers(data: CalculateInput) -> str:
    """Subtract two numbers."""
    result = data.a - data.b
    return f'The difference of {data.a} and {data.b} is {result}'


@reasoning_tool
def get_weather(data: WeatherInput) -> str:
    """Get weather information for a location (mock implementation)."""
    return f'The weather in {data.location} is sunny with a temperature of 72°F'


@reasoning_tool
def get_news(data: NewsInput) -> str:
    """Get news about a topic (mock implementation)."""
    return f'Latest news about {data.topic}: Sample news article content here'


@reasoning_tool
def calculate_square(data: CalculateInput) -> str:
    """Calculate the square of a number (uses only 'a' parameter)."""
    result = data.a**2
    return f'The square of {data.a} is {result}'


def example_1_basic_selection() -> None:
    """Example 1: Let LLM select relevant tools from a large set."""
    print('\nEXAMPLE 1: Basic Tool Selection')
    print('8 tools available, LLM selects only relevant ones\n')

    selector = TinyLLMToolSelectorMiddlewareConfig(
        llm=build_llm('openai:gpt-4o-mini'),
    )

    agent = build_agent(
        'multistep',
        llm='openai:gpt-4o-mini',
        tools=[
            greet,
            add_numbers,
            multiply_numbers,
            divide_numbers,
            subtract_numbers,
            get_weather,
            get_news,
            calculate_square,
        ],
        middleware=[selector],
    )

    result = agent.run('Greet Alice and then add 5 and 7')
    print(f'Result: {result}\n')


def example_2_max_tools_limit() -> None:
    """Example 2: Limit maximum number of tools selected."""
    print('\nEXAMPLE 2: Limit Maximum Tools')
    print('8 tools available, max_tools=3\n')

    selector = build_middleware(
        'llm_tool_selector', llm=build_llm('openai:gpt-4o-mini'), max_tools=3
    )

    agent = build_agent(
        'multistep',
        llm='openai:gpt-4o-mini',
        tools=[
            greet,
            add_numbers,
            multiply_numbers,
            divide_numbers,
            subtract_numbers,
            get_weather,
            get_news,
            calculate_square,
        ],
        middleware=[selector],
    )

    result = agent.run(
        'Calculate: 5+7, 3*4, and 10-2. Also greet Bob and get weather for NYC'
    )
    print(f'Result: {result}\n')


def example_3_always_include() -> None:
    """Example 3: Always include specific tools regardless of relevance."""
    print('\nEXAMPLE 3: Always Include Specific Tools')
    print('Always include "greet" tool + dynamic selection\n')

    selector = build_middleware(
        'llm_tool_selector',
        llm=build_llm('openai:gpt-4o-mini'),
        always_include=['greet'],
    )

    agent = build_agent(
        'multistep',
        llm='openai:gpt-4o-mini',
        tools=[
            greet,
            add_numbers,
            multiply_numbers,
            divide_numbers,
            subtract_numbers,
            get_weather,
            get_news,
            calculate_square,
        ],
        middleware=[selector],
    )

    result = agent.run('Calculate 5 + 7 and then multiply the result by 3')
    print(f'Result: {result}\n')


def example_4_combined_constraints() -> None:
    """Example 4: Combine max_tools and always_include."""
    print('\nEXAMPLE 4: Combined Constraints')
    print('Always include "greet" + max 4 tools total\n')

    selector = TinyLLMToolSelectorMiddlewareConfig(
        llm=build_llm('openai:gpt-4o-mini'),
        max_tools=4,
        always_include=['greet'],
    )

    agent = build_agent(
        'multistep',
        llm='openai:gpt-4o-mini',
        tools=[
            greet,
            add_numbers,
            multiply_numbers,
            divide_numbers,
            subtract_numbers,
            get_weather,
            get_news,
            calculate_square,
        ],
        middleware=[selector],
    )

    result = agent.run(
        'Greet Alice, calculate 5+7 and 3*4, get weather for Paris, '
        'and find news about AI'
    )
    print(f'Result: {result}\n')


def main() -> None:
    print('\n=== LLM TOOL SELECTOR MIDDLEWARE EXAMPLES ===')
    print('Intelligently selects relevant tools to reduce token usage\n')

    example_1_basic_selection()
    example_2_max_tools_limit()
    example_3_always_include()
    example_4_combined_constraints()

    print('=== ALL EXAMPLES COMPLETED ===\n')


if __name__ == '__main__':
    main()
