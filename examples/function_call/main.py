from tinygent.tools.tool import tool


@tool
def add(a: int, b: int = 3) -> int:
    """Adds two numbers."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtracts two numbers."""
    return a - b


if __name__ == '__main__':
    add.info.print_summary()

    schema = add.info.input_schema
    if schema is not None:
        for name, field in schema.model_fields.items():
            print(f"{name}: {field}, default={field.default}")
