from tinygent.tools.tool import tool


@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtracts two numbers."""
    return a - b


if __name__ == '__main__':
    print(add.info.arg_count)
