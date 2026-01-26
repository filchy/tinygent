import logging

from pydantic import Field

from tinygent.agents.middleware.tool_limiter import TinyToolCallLimiterMiddleware
from tinygent.agents.multi_step_agent import TinyMultiStepAgent
from tinygent.core.factory import build_llm
from tinygent.core.prompts.agents.template.multi_agent import ActionPromptTemplate
from tinygent.core.prompts.agents.template.multi_agent import (
    FallbackAnswerPromptTemplate,
)
from tinygent.core.prompts.agents.template.multi_agent import MultiStepPromptTemplate
from tinygent.core.prompts.agents.template.multi_agent import PlanPromptTemplate
from tinygent.core.types.base import TinyModel
from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.tools import reasoning_tool

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s',
)


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


plan_prompt = PlanPromptTemplate(
    init_plan='Plan for solving: {{ task }} with tools: {{ tools }}',
    update_plan=(
        'Update plan for: {{ task }} '
        'using tools: {{ tools }}, '
        'history: {{ history }}, '
        'steps so far: {{ steps }}, '
        'remaining: {{ remaining_steps }}'
    ),
)

action_prompt = ActionPromptTemplate(
    system='You are a helpful agent. Use the available tools to complete tasks.',
    final_answer=(
        'Solve task: {{ task }} using steps {{ steps }}, tools {{ tools }}, '
        'and tool calls {{ tool_calls }}. '
        'Conversation so far: {{ history }}'
    ),
)

fallback_prompt = FallbackAnswerPromptTemplate(
    fallback_answer=(
        'Provide the final answer for {{ task }} '
        'based on history: {{ history }} and steps {{ steps }}'
    )
)

prompt_template = MultiStepPromptTemplate(
    plan=plan_prompt,
    acter=action_prompt,
    fallback=fallback_prompt,
)


def example_1_global_limit() -> None:
    """Example 1: Limit all tools globally."""
    print('\n' + '=' * 70)
    print('EXAMPLE 1: Global Tool Call Limit')
    print('=' * 70)
    print('Limit all tools to 3 calls total')
    print('Task requires 4 tool calls, agent will finish with available info')
    print('=' * 70 + '\n')

    limiter = TinyToolCallLimiterMiddleware(max_tool_calls=3)

    agent = TinyMultiStepAgent(
        llm=build_llm('openai:gpt-4o-mini'),
        prompt_template=prompt_template,
        tools=[greet, add_numbers, multiply_numbers, divide_numbers],
        max_iterations=10,
        middleware=[limiter],
        memory=BufferChatMemory(),
    )

    result = agent.run('Greet Alice, add 5 and 7, multiply 3 and 4, then greet Bob')

    print('\n' + '=' * 70)
    print('Result:', result)
    print('=' * 70)

    print('\nLimiter Stats:')
    stats = limiter.get_stats()
    for key, value in stats.items():
        print(f'  {key}: {value}')


def example_2_specific_tool_limit() -> None:
    """Example 2: Limit only specific tool."""
    print('\n' + '=' * 70)
    print('EXAMPLE 2: Limit Specific Tool Only')
    print('=' * 70)
    print('Limit only "greet" tool to 1 call')
    print('Other tools can be called unlimited times')
    print('=' * 70 + '\n')

    greet_limiter = TinyToolCallLimiterMiddleware(tool_name='greet', max_tool_calls=1)

    agent = TinyMultiStepAgent(
        llm=build_llm('openai:gpt-4o-mini'),
        prompt_template=prompt_template,
        tools=[greet, add_numbers, multiply_numbers, divide_numbers],
        max_iterations=10,
        middleware=[greet_limiter],
        memory=BufferChatMemory(),
    )

    result = agent.run(
        'Greet Alice, then do these calculations: add 5+7, multiply 3*4, divide 10/2'
    )

    print('\n' + '=' * 70)
    print('Result:', result)
    print('=' * 70)

    print('\nLimiter Stats:')
    stats = greet_limiter.get_stats()
    for key, value in stats.items():
        print(f'  {key}: {value}')


def example_3_multiple_limiters() -> None:
    """Example 3: Multiple limiters for different tools."""
    print('\n' + '=' * 70)
    print('EXAMPLE 3: Multiple Limiters for Different Tools')
    print('=' * 70)
    print('Limit greet to 1 call, math operations to 2 calls each')
    print('=' * 70 + '\n')

    middleware = [
        TinyToolCallLimiterMiddleware(tool_name='greet', max_tool_calls=1),
        TinyToolCallLimiterMiddleware(tool_name='add_numbers', max_tool_calls=2),
        TinyToolCallLimiterMiddleware(tool_name='multiply_numbers', max_tool_calls=2),
    ]

    agent = TinyMultiStepAgent(
        llm=build_llm('openai:gpt-4o-mini'),
        prompt_template=prompt_template,
        tools=[greet, add_numbers, multiply_numbers, divide_numbers],
        max_iterations=10,
        middleware=middleware,
        memory=BufferChatMemory(),
    )

    result = agent.run('Greet Alice and Bob. Calculate: 5+7, 10+20, 3*4, 6*8')

    print('\n' + '=' * 70)
    print('Result:', result)
    print('=' * 70)

    print('\nAll Limiter Stats:')
    for i, limiter in enumerate(middleware, 1):
        print(f'\n  Limiter {i}:')
        stats = limiter.get_stats()
        for key, value in stats.items():
            print(f'    {key}: {value}')


def main() -> None:
    print('\n')
    print('*' * 70)
    print('TOOL CALL LIMITER MIDDLEWARE EXAMPLES')
    print('*' * 70)

    example_1_global_limit()
    example_2_specific_tool_limit()
    example_3_multiple_limiters()

    print('\n')
    print('*' * 70)
    print('ALL EXAMPLES COMPLETED')
    print('*' * 70)
    print('\n')


if __name__ == '__main__':
    main()
