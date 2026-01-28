from pydantic import Field

from tinygent.agents.middleware import TinyToolCallLimiterMiddleware
from tinygent.core.factory import build_agent
from tinygent.core.types import TinyModel
from tinygent.logging import setup_logger
from tinygent.tools import reasoning_tool

logging = setup_logger('debug')


class GreetInput(TinyModel):
    name: str = Field(..., description='The name of the person to greet.')


class CalculateInput(TinyModel):
    a: int = Field(..., description='First number')
    b: int = Field(..., description='Second number')


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


def example_1_global_limit() -> None:
    """Example 1: Limit all tools globally."""
    print('\nEXAMPLE 1: Global Tool Call Limit')
    print('Limit all tools to 3 calls total\n')

    limiter = TinyToolCallLimiterMiddleware(max_tool_calls=3)

    agent = build_agent(
        'multistep',
        llm='openai:gpt-4o-mini',
        tools=[greet, add_numbers, multiply_numbers, divide_numbers],
        middleware=[limiter],
    )

    result = agent.run('Greet Alice, add 5 and 7, multiply 3 and 4, then greet Bob')
    print(f'Result: {result}')

    print('\nLimiter Stats:')
    for key, value in limiter.get_stats().items():
        print(f'\t{key}: {value}')
    print()


def example_2_specific_tool_limit() -> None:
    """Example 2: Limit only specific tool."""
    print('\nEXAMPLE 2: Limit Specific Tool Only')
    print('Limit "greet" tool to 1 call, others unlimited\n')

    greet_limiter = TinyToolCallLimiterMiddleware(tool_name='greet', max_tool_calls=1)

    agent = build_agent(
        'multistep',
        llm='openai:gpt-4o-mini',
        tools=[greet, add_numbers, multiply_numbers, divide_numbers],
        middleware=[greet_limiter],
    )

    result = agent.run(
        'Greet Alice, then do these calculations: add 5+7, multiply 3*4, divide 10/2'
    )
    print(f'Result: {result}')

    print('\nLimiter Stats:')
    for key, value in greet_limiter.get_stats().items():
        print(f'\t{key}: {value}')
    print()


def example_3_multiple_limiters() -> None:
    """Example 3: Multiple limiters for different tools."""
    print('\nEXAMPLE 3: Multiple Limiters for Different Tools')
    print('Limit greet to 1 call, math operations to 2 calls each\n')

    middleware = [
        TinyToolCallLimiterMiddleware(tool_name='greet', max_tool_calls=1),
        TinyToolCallLimiterMiddleware(tool_name='add_numbers', max_tool_calls=2),
        TinyToolCallLimiterMiddleware(tool_name='multiply_numbers', max_tool_calls=2),
    ]

    agent = build_agent(
        'multistep',
        llm='openai:gpt-4o-mini',
        tools=[greet, add_numbers, multiply_numbers, divide_numbers],
        middleware=middleware,
    )

    result = agent.run('Greet Alice and Bob. Calculate: 5+7, 10+20, 3*4, 6*8')
    print(f'Result: {result}')

    print('\nAll Limiter Stats:')
    for i, limiter in enumerate(middleware, 1):
        print(f'\tLimiter {i}:')
        for key, value in limiter.get_stats().items():
            print(f'\t\t{key}: {value}')
    print()


def main() -> None:
    print('\n=== TOOL CALL LIMITER MIDDLEWARE EXAMPLES ===\n')

    example_1_global_limit()
    example_2_specific_tool_limit()
    example_3_multiple_limiters()

    print('=== ALL EXAMPLES COMPLETED ===\n')


if __name__ == '__main__':
    main()
