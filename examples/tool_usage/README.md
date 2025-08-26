# Tool Example

This example demonstrates how to use the `@tool` decorator from **tinygent** to wrap various types of Python functions (sync, async, generators) into a unified execution interface.

---

## Requirements

Each tool **must accept zero or one argument**, and if it does accept an argument, it must be a subclass of `pydantic.BaseModel`.

This design allows:

1. **Parameter validation & introspection**
   Input parameters are modeled as Pydantic fields, providing type checking and field descriptions (via `Field(...)`).

2. **LLM-compatible tool definitions**
   Tools can be described with OpenAI-compatible schemas and used directly for function-calling with models.

---

### Example Input Schema

```python
from pydantic import BaseModel, Field

class AddInput(BaseModel):
    a: int = Field(..., description="First number to add")
    b: int = Field(..., description="Second number to add")
```

---

## Features

Each decorated function becomes a `Tool` instance that:

* Exposes a unified `.run()` and `__call__()` interface
* Supports the following function types:

  * Sync function
  * Async coroutine
  * Sync generator
  * Async generator
* Automatically parses:

  * Pydantic model instances
  * Plain dicts
  * `**kwargs`
  * `*args` (including a single dict as positional input)
* Provides full metadata via `ToolInfo`

---

## `.run()` and `__call__()` Behavior

Internally, both `.run()` and `__call__()` support the same logic:

* If input schema is present:

  * `dict` or `**kwargs` will be validated and parsed into the appropriate `BaseModel`
  * A single validated model instance is passed through unchanged
* Tools with **no inputs** work as well

Execution logic:

* Async coroutine: awaited
* Generator: iterated and collected
* Async generator: asynchronously iterated and collected
* Sync function: called directly and returned

---

## Included Examples

```python
@tool
def add(data: AddInput) -> int:
    return data.a + data.b


@tool
async def greet(data: GreetInput) -> str:
    return f'Hello, {data.name}!'


@tool
def count(data: CountInput):
    for i in range(1, data.n + 1):
        yield i


@tool
async def async_count(data: CountInput):
    for i in range(1, data.n + 1):
        yield i
```

---

## Calling Examples

```python
# BaseModel instance
print(add(AddInput(a=1, b=2)))

# Dict input
print(greet({"name": "TinyGent"}))

# Kwargs input
print(list(count.run(n=3)))

# Positional dict (args)
print(list(async_count.run({"n": 4})))
```

Each call style is automatically parsed and dispatched based on the function type and the tool's schema.

---

## Running the Example

```bash
uv run main.py
```

Expected output:

```
3
Hello, TinyGent!
[1, 2, 3]
[1, 2, 3, 4]
```

---

## Use Cases

* LLM-based tool/function calling
* Multi-agent orchestration
* Declarative plugin registration with type-safe inputs
