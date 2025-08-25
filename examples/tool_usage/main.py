from tinygent.tools.tool import tool


@tool
def add(a: int, b: int) -> int:

    return a + b


@tool
async def greet(name: str) -> str:

    return f'Hello, {name}'


@tool
def count(n: int):

    for i in range(1, n + 1):
        yield i


@tool
async def async_count(n: int):

    for i in range(1, n + 1):
        yield i


if __name__ == '__main__':
    print(add.run(1, 2))
    print(greet.run('TinyGent'))
    print(count.run(3))
    print(async_count.run(3))
