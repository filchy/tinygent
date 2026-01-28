from pydantic import Field

from tinygent.core.runtime.tool_catalog import GlobalToolCatalog
from tinygent.core.types import TinyModel
from tinygent.tools import jit_tool
from tinygent.tools import register_reasoning_tool
from tinygent.tools import register_tool
from tinygent.tools import tool


class AddInput(TinyModel):
    a: int = Field(..., description='The first number to add.')
    b: int = Field(..., description='The second number to add.')


@register_tool(use_cache=True)
def add(data: AddInput) -> int:
    """Adds two numbers together."""

    return data.a + data.b


# Variant 2: Regular parameters (auto-generated schema)
# No need to define a TinyModel class - schema is auto-generated from type hints.
@register_tool(use_cache=True)
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together."""

    return a * b


class GreetInput(TinyModel):
    name: str = Field(..., description='The name to greet.')


@register_tool(use_cache=True)
async def greet(data: GreetInput) -> str:
    """Greets a person by name."""

    return f'Hello, {data.name}!'


class CountInput(TinyModel):
    n: int = Field(..., description='The number to count to.')


@jit_tool(jit_instruction='Count from 1 to n, yielding each number.')
def count(data: CountInput):
    """Counts from 1 to n, yielding each number."""

    for i in range(1, data.n + 1):
        yield i


class AsyncCountInput(TinyModel):
    n: int = Field(..., description='The number to count to.')


@tool
async def async_count(data: AsyncCountInput):
    """Asynchronously counts from 1 to n, yielding each number."""

    for i in range(1, data.n + 1):
        yield i


# Another example with @tool decorator using regular parameters
@tool
def divide(a: float, b: float) -> float:
    """Divides a by b."""

    if b == 0:
        return float('inf')
    return a / b


class SearchInput(TinyModel):
    query: str = Field(..., description='Search query')


@register_reasoning_tool(reasoning_prompt='Explain why you are performing this search.')
def search(data: SearchInput) -> str:
    """Search for something."""
    return f'Results for {data.query}'


if __name__ == '__main__':
    header_print = lambda title: print('\n' + '*' * 10 + f' {title} ' + '*' * 10 + '\n')
    classic_print = lambda msg: print(f'[Classic] {msg}')
    global_registry_print = lambda msg: print(f'[GlobalToolCatalog] {msg}')
    cache_print = lambda msg: print(f'[Cache] {msg}')

    # Tool summaries
    header_print('Tool Summaries')

    add.info.print_summary()
    multiply.info.print_summary()  # Regular params variant
    greet.info.print_summary()
    count.info.print_summary()
    async_count.info.print_summary()
    divide.info.print_summary()  # Regular params variant

    # Execute the tools directly
    header_print('Direct Executions (TinyModel variant)')

    classic_print(add(AddInput(a=1, b=2)))

    classic_print(greet({'name': 'TinyGent'}))

    classic_print(list(count(n=3)))

    classic_print(list(async_count({'n': 4})))

    # Execute tools with regular params variant
    header_print('Direct Executions (Regular params variant)')

    classic_print(multiply(a=3, b=4))  # kwargs
    classic_print(multiply({'a': 5, 'b': 6}))  # dict
    classic_print(divide(a=10, b=3))  # kwargs
    classic_print(divide({'a': 20, 'b': 4}))  # dict

    # Global registry
    header_print('Global Registry Executions')

    registry = GlobalToolCatalog().get_active_catalog()

    registry_add = registry.get_tool('add')
    global_registry_print(registry_add(a=1, b=2))

    registry_greet = registry.get_tool('greet')
    global_registry_print(registry_greet({'name': 'TinyGent'}))

    header_print('Local Tool Executions')

    classic_print(list(count(n=5)))
    classic_print(list(async_count({'n': 6})))

    # Cache info
    header_print('Cache Info')

    cache_print(add.cache_info())
    cache_print(greet.cache_info())
    cache_print(count.cache_info())
    cache_print(async_count.cache_info())

    # Clear caches
    header_print('Clear Caches')
    add.clear_cache()
    greet.clear_cache()
    count.clear_cache()
    async_count.clear_cache()

    cache_print(add.cache_info())
    cache_print(greet.cache_info())
    cache_print(count.cache_info())
    cache_print(async_count.cache_info())

    # Reasoning tool
    header_print('Reasoning Tool Execution')

    global_registry_search = registry.get_tool('search')
    global_registry_print(global_registry_search({'query': 'TinyGent'}))

    # NOTE: count and async_count are not cachable, so their cache_info will be None
