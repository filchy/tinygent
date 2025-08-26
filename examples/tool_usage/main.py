from pydantic import Field
from pydantic import BaseModel

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
    print(add.run(1, 2))
    print(greet.run('TinyGent'))
    print(count.run(3))
    print(async_count.run(3))
