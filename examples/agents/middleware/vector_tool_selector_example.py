from typing import TypeGuard

from pydantic import Field

from tinygent.agents.middleware.vector_tool_selector import (
    TinyVectorToolSelectorMiddlewareConfig,
)
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.datamodels.tool import AbstractToolConfig
from tinygent.core.factory import build_agent
from tinygent.core.factory import build_embedder
from tinygent.core.factory import build_middleware
from tinygent.core.types import TinyModel
from tinygent.logging import setup_logger
from tinygent.tools import reasoning_tool

logging = setup_logger('info')


class GreetInput(TinyModel):
    name: str = Field(..., description='The name of the person to greet.')


class CalculateInput(TinyModel):
    a: int = Field(..., description='First number')
    b: int = Field(..., description='Second number')


class WeatherInput(TinyModel):
    location: str = Field(..., description='Location to get weather for')


class NewsInput(TinyModel):
    topic: str = Field(..., description='News topic to search for')


class TranslateInput(TinyModel):
    text: str = Field(..., description='Text to translate')
    language: str = Field(..., description='Target language')


class SummarizeInput(TinyModel):
    text: str = Field(..., description='Text to summarize')


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
    return f'The weather in {data.location} is sunny with a temperature of 72F'


@reasoning_tool
def get_news(data: NewsInput) -> str:
    """Get news about a topic (mock implementation)."""
    return f'Latest news about {data.topic}: Sample news article content here'


@reasoning_tool
def translate_text(data: TranslateInput) -> str:
    """Translate text to a target language (mock implementation)."""
    return f'[{data.language}] {data.text}'


@reasoning_tool
def summarize_text(data: SummarizeInput) -> str:
    """Summarize a long piece of text (mock implementation)."""
    return f'Summary: {data.text[:50]}...'


ALL_TOOLS: list[dict | AbstractTool | AbstractToolConfig | str] = [
    greet,
    add_numbers,
    multiply_numbers,
    divide_numbers,
    subtract_numbers,
    get_weather,
    get_news,
    translate_text,
    summarize_text,
]


def example_1_basic_selection() -> None:
    """Example 1: Use embeddings to select relevant tools from a large set."""
    print('\nEXAMPLE 1: Basic Vector Tool Selection')
    print('9 tools available, embedder selects only semantically relevant ones\n')

    selector = TinyVectorToolSelectorMiddlewareConfig(
        embedder=build_embedder('openai:text-embedding-3-small'),
        similarity_threshold=0.1,
    )

    agent = build_agent(
        'multistep',
        llm='openai:gpt-4o-mini',
        tools=ALL_TOOLS,
        middleware=[selector],
    )

    result = agent.run(
        'Greet Alice and then tell her what is weather like in San Francisco'
    )
    print(f'Result: {result}\n')


def example_2_max_tools_limit() -> None:
    """Example 2: Limit maximum number of tools selected by cosine similarity."""
    print('\nEXAMPLE 2: Limit Maximum Tools')
    print('9 tools available, max_tools=3\n')

    selector = build_middleware(
        'vector_tool_classifier',
        embedder=build_embedder('openai:text-embedding-3-small'),
        max_tools=3,
    )

    agent = build_agent(
        'multistep',
        llm='openai:gpt-4o-mini',
        tools=ALL_TOOLS,
        middleware=[selector],
    )

    result = agent.run('What is 15 divided by 3? Then multiply the result by 4.')
    print(f'Result: {result}\n')


def example_3_always_include() -> None:
    """Example 3: Always include specific tools regardless of similarity score."""
    print('\nEXAMPLE 3: Always Include Specific Tools')
    print('Always include "greet" tool + vector selection for the rest\n')

    selector = build_middleware(
        'vector_tool_classifier',
        embedder=build_embedder('openai:text-embedding-3-small'),
        similarity_threshold=0.1,
        always_include=[greet],
        max_tools=3,
    )

    agent = build_agent(
        'multistep',
        llm='openai:gpt-4o-mini',
        tools=ALL_TOOLS,
        middleware=[selector],
    )

    result = agent.run(
        'Summarize the following: "The quick brown fox jumps over the lazy dog."'
    )
    print(f'Result: {result}\n')


def example_4_custom_transform_fns() -> None:
    """Example 4: Custom query and tool transform functions."""
    print('\nEXAMPLE 4: Custom Transform Functions')
    print('Custom functions control what gets embedded for query and tools\n')

    from tinygent.agents.middleware.vector_tool_selector import (
        TinyVectorToolSelectorMiddleware,
    )
    from tinygent.core.datamodels.tool import AbstractTool
    from tinygent.core.types.io.llm_io_input import TinyLLMInput

    class HasContent:
        content: str

    def _has_content(m: object) -> TypeGuard['HasContent']:
        return hasattr(m, 'content')

    def query_transform(llm_input: TinyLLMInput) -> str:
        # Embed only the last 3 messages combined instead of just the last human message

        recent = (
            llm_input.messages[-3:]
            if len(llm_input.messages) >= 3
            else llm_input.messages
        )
        return ' '.join(m.content for m in recent if _has_content(m))

    def tool_transform(tool: AbstractTool) -> str:
        # Include tool name twice to give it more weight in the embedding
        return f'{tool.info.name} {tool.info.name}: {tool.info.description}'

    selector = TinyVectorToolSelectorMiddleware(
        embedder=build_embedder('openai:text-embedding-3-small'),
        similarity_threshold=0.1,
        max_tools=4,
        query_transform_fn=query_transform,
        tool_transform_fn=tool_transform,
    )

    agent = build_agent(
        'multistep',
        llm='openai:gpt-4o-mini',
        tools=ALL_TOOLS,
        middleware=[selector],
    )

    result = agent.run('Translate "Hello world" to Spanish')
    print(f'Result: {result}\n')


def main() -> None:
    print('\n=== VECTOR TOOL SELECTOR MIDDLEWARE EXAMPLES ===')
    print('Selects relevant tools using semantic similarity (embeddings)\n')

    example_1_basic_selection()
    example_2_max_tools_limit()
    example_3_always_include()
    example_4_custom_transform_fns()

    print('=== ALL EXAMPLES COMPLETED ===\n')


if __name__ == '__main__':
    main()
