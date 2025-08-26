# Tool Example: Unified `.run()` for Sync, Async, Generator, Async Generator

This example demonstrates how to use the `@tool` decorator from **tinygent**
to wrap various types of Python functions (sync, async, generators) into a unified execution interface.

---

## Requirements

Each tool **must accept a single input argument**, which is a subclass of `pydantic.BaseModel`.

This serves two main purposes:

1. **Parameter validation & documentation**
   Each field in the `BaseModel` represents one input parameter, with full type checking and optional descriptions via `Field(...)`.

2. **LLM-compatible function calling**
   The structure allows for automatic generation of OpenAI-compatible tool schemas.
   This is necessary when exposing tools to language models via function-calling APIs so the model can understand parameter names, types, and descriptions.

### Example Input Schema

```python
from pydantic import BaseModel, Field

class AddInput(BaseModel):
    a: int = Field(..., description="First number to add")
    b: int = Field(..., description="Second number to add")
```

---

## Features

Each decorated function is converted into a `Tool` instance that:

* Exposes a unified `.run()` method for all function types
* Supports sync, async, generator, and async generator functions
* Automatically detects function type (coroutine, generator, etc.)
* Stores metadata using `ToolInfo` (input schema, async mode, etc.)

---

## Behavior of `.run()`

* Awaits async coroutines
* Iterates and collects items from generators and async generators
* Returns raw values for plain sync functions

This is useful for:

* LLM function-calling / plugin systems
* Multi-agent tool orchestration
* Dynamic tool registry and execution

---

## Included Examples

```python
@tool
def add(data: AddInput) -> int:
    return data.a + data.b


@tool
async def greet(data: GreetInput) -> str:
    return f'Hello, {data.name}'


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

## Running the Example

```bash
uv run main.py
```

Expected output:

```
3
Hello, TinyGent
[1, 2, 3]
[1, 2, 3]
```
