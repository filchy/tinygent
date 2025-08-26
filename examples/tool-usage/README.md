# Tool Example

This example demonstrates how to use the `@tool` decorator from **tinygent** to wrap various types of Python functions (sync, async, generators) into a unified execution interface. Every decorated function is automatically registered into the **global runtime tool registry**, making it instantly accessible via `GlobalRegistry.get_registry().get_tool('<name>')`.

---

## Requirements

Each tool **must accept zero or one argument**, and if it does accept an argument, it must be a subclass of `pydantic.BaseModel`.

This design allows:

1. **Parameter validation & introspection**
   Input parameters are modeled as Pydantic fields, providing type checking and field descriptions (via `Field(...)`).

2. **LLM-compatible tool definitions**
   Tools can be described with OpenAI-compatible schemas and used directly for function-calling with models.

3. **Auto-registration in the runtime registry**
   Each use of `@tool` not only wraps the function, but also **registers it automatically** into the global registry. This means you can dynamically retrieve any tool by name at runtime.

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

* Exposes a unified `__call__()` interface
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
* Registers itself into the `GlobalRegistry` under its function name
* Provides full metadata via `ToolInfo`

---

## Tool Description

The tool's **description** is automatically extracted from the function's **docstring**.

This is used for introspection, OpenAI-compatible tool schemas, and registry summaries.

```python
@tool
def add(data: AddInput) -> int:
    """Adds two numbers together."""
    return data.a + data.b
```

In this case, the description for the `add` tool will be `"Adds two numbers together."`.

Writing clear and concise docstrings is essential, as this metadata is often used in LLM-assisted reasoning and tool selection.

---

## `__call__()` Behavior

The public interface for tools is the `__call__()` method. It supports:

* BaseModel input
* Raw `dict` input (validated)
* `**kwargs` input (validated)
* Positional `*args` (e.g., one dict argument)

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

All of the above tools are automatically registered and retrievable from the global registry using their function names.

---

## Calling Examples

```python
# BaseModel instance
print(add(AddInput(a=1, b=2)))

# Dict input
print(greet({"name": "TinyGent"}))

# Kwargs input
print(list(count(n=3)))

# Positional dict (args)
print(list(async_count({"n": 4})))

# Access from global registry
from tinygent.runtime.global_registry import GlobalRegistry
registry = GlobalRegistry.get_registry()

greet_tool = registry.get_tool("greet")
print(greet_tool(name="TinyGent"))
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
Hello, TinyGent!
```
