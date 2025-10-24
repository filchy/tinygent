# Tool Example

This example demonstrates how to use the `@tool` and `@register_tool` decorators from **tinygent**.

* **`@tool`** wraps a function into a `Tool` object (unified interface, metadata, schema validation).
* **`@register_tool`** does the same, but also **registers it automatically** into the global runtime tool registry, making it instantly accessible via `GlobalRegistry.get_registry().get_tool('<name>')`.

---

## Requirements

Each tool **must accept zero or one argument**, and if it does accept an argument, it must be a subclass of `tinygent.types.TinyModel`.

This design allows:

1. **Schema validation** — arguments validated against Pydantic (`TinyModel`).
2. **LLM compatibility** — tools have OpenAI-compatible schemas for function calling.
3. **Optional auto-registration** — tools are globally available if you use `@register_tool`.

---

### Example Input Schema

```python
from pydantic import Field
from tinygent.types.base import TinyModel

class AddInput(TinyModel):
    a: int = Field(..., description='First number to add')
    b: int = Field(..., description='Second number to add')
```

---

## Decorators

### `@tool`

Creates a `Tool` instance but does **not** register it.

```python
from tinygent.tools.tool import tool

@tool
def local_add(data: AddInput) -> int:
    """Adds two numbers."""
    return data.a + data.b

print(local_add(AddInput(a=1, b=2)))  # works directly
```

### `@register_tool`

Creates a `Tool` instance **and registers it** into the global registry.

```python
from tinygent.tools.tool import register_tool

@register_tool(use_cache=True)
def add(data: AddInput) -> int:
    """Adds two numbers together."""
    return data.a + data.b

from tinygent.runtime.global_registry import GlobalRegistry

registry = GlobalRegistry.get_registry()
add_tool = registry.get_tool('add')
print(add_tool(a=1, b=2))
```

---

## Tool Features

Each decorated function becomes a `Tool` instance that:

* Exposes a unified `__call__()` interface
* Supports:

  * Sync function
  * Async coroutine
  * Sync generator
  * Async generator
* Accepts input as:

  * `TinyModel` instance
  * Raw `dict` (validated)
  * `**kwargs` (validated)
  * Positional dict (`*args`)
* Provides full metadata via `ToolInfo`

---

## Caching Support

You can enable in-memory LRU caching for sync/async tools with `use_cache=True`:

```python
@register_tool(use_cache=True, cache_size=256)
def expensive_tool(data: AddInput) -> int:
    return data.a + data.b

print(expensive_tool(AddInput(a=1, b=2)))
print(expensive_tool.cache_info())
```

* Uses `functools.lru_cache` (sync) or `async_lru.alru_cache` (async)
* Not supported for generators
* Cache inspection and clearing:

```python
expensive_tool.cache_info()
expensive_tool.clear_cache()
```

---

## Calling Examples

```python
# TinyModel instance
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

print(registry.get_tool("greet")({"name": "TinyGent"}))
```

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
