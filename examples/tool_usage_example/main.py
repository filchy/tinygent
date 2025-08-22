"""
Tool Example: Demonstrating Sync, Async, Generator, and Async Generator

This file showcases how to use the `@tool` decorator from tinygent
to automatically wrap different types of functions (sync, async,
generator, async generator) into a unified interface.

Each decorated function is converted into a `Tool` instance that:
- Inspects and validates its input and output types.
- Automatically generates a Pydantic input/output schema.
- Provides a `.run()` method that works uniformly across all function types.

The `.run()` method:
- Awaits async coroutines
- Iterates and collects results from generators and async generators
- Returns plain values for sync functions

This is useful for building LLM toolchains, plugin systems, or
general-purpose dynamic function execution.
"""

from tinygent.tools.tool import tool


@tool
def add(a: int, b: int) -> int:

    return a + b


@tool
async def greet(name: str) -> str:

    return f"Hello, {name}"


@tool
def count(n: int):

    for i in range(1, n + 1):
        yield i


@tool
async def async_count(n: int):

    for i in range(1, n + 1):
        yield i


if __name__ == "__main__":
    print(add.run(1, 2))
    print(greet.run("Filip"))
    print(count.run(3))
    print(async_count.run(3))
