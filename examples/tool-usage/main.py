from pydantic import BaseModel
from pydantic import Field

from tinygent.tools.tool import tool


class AddInput(BaseModel):
    a: int = Field(..., description='The first number to add.')
    b: int = Field(..., description='The second number to add.')


@tool
def add(data: AddInput) -> int:
    return data.a + data.b


class GreetInput(BaseModel):
    name: str = Field(..., description='The name to greet.')


@tool
async def greet(data: GreetInput) -> str:
    return f'Hello, {data.name}!'


class CountInput(BaseModel):
    n: int = Field(..., description='The number to count to.')


@tool
def count(data: CountInput):
    for i in range(1, data.n + 1):
        yield i


class AsyncCountInput(BaseModel):
    n: int = Field(..., description='The number to count to.')


@tool
async def async_count(data: AsyncCountInput):
    for i in range(1, data.n + 1):
        yield i


if __name__ == '__main__':
    classic_print = lambda msg: print(f'[Classic] {msg}')
    global_registry_print = lambda msg: print(f'[GlobalRegistry] {msg}')

    # Execute the tools directly
    classic_print(add(AddInput(a=1, b=2)))

    classic_print(greet({'name': 'TinyGent'}))

    classic_print(list(count(n=3)))

    classic_print(list(async_count({'n': 4})))

    # Global registry
    from tinygent.runtime.global_registry import GlobalRegistry

    registry = GlobalRegistry.get_registry()

    registry_add = registry.get_tool('add')
    global_registry_print(registry_add(a=1, b=2))

    registry_greet = registry.get_tool('greet')
    global_registry_print(registry_greet({'name': 'TinyGent'}))

    registry_count = registry.get_tool('count')
    global_registry_print(list(registry_count(n=3)))

    registry_async_count = registry.get_tool('async_count')
    global_registry_print(list(registry_async_count({'n': 4})))
