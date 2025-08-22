import inspect
from typing import Any

from tinygent.tools.tool import Tool
from tinygent.tools.tool import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""

    return a + b


@tool
async def async_hello(name: str) -> str:
    """Return a greeting message."""

    return f"Hello, {name}!"


@tool
def count_up_to(n: int):
    """Count up to a number."""

    for i in range(1, n + 1):
        yield i


@tool
async def async_count_up_to(n: int):
    """Asynchronously count up to a number."""

    for i in range(1, n + 1):
        yield i


def run_tool_minimal(t: Tool, *args: Any, **kwargs: Any) -> None:

    t.info.print_summary()
    print("Result:")

    if t.info.is_async_generator:
        async def runner():
            async for value in t(*args, **kwargs):
                print(value)

        import asyncio
        asyncio.run(runner())

    elif t.info.is_coroutine:
        async def runner():
            result = await t(*args, **kwargs)
            print(result)

        import asyncio
        asyncio.run(runner())

    else:
        result = t(*args, **kwargs)
        for val in result if inspect.isgenerator(result) else [result]:
            print(val)


if __name__ == "__main__":
    run_tool_minimal(add, 3, 5)
    run_tool_minimal(async_hello, "Alice")
    run_tool_minimal(count_up_to, 5)
    run_tool_minimal(async_count_up_to, 5)
