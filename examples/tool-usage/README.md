# Tool Example

This example demonstrates how to use the `@tool` and `@register_tool` decorators from **tinygent**.

* **`@tool`** wraps a function into a `Tool` object (unified interface, metadata, schema validation).
* **`@register_tool`** does the same, but also **registers it automatically** into the global runtime tool registry catalog, making it instantly accessible via `GlobalToolCatalog().get_active_catalog().get_tool('<name>')`.

---

## Tool Parameter Variants

TinyGent supports **two ways** to define tool parameters:

### Variant 1: TinyModel Descriptor (Explicit Schema)

Pass a single `TinyModel` subclass as the only argument. This gives you full control over field descriptions and validation.

```python
from pydantic import Field
from tinygent.core.types import TinyModel
from tinygent.tools import tool

class AddInput(TinyModel):
    a: int = Field(..., description='First number to add')
    b: int = Field(..., description='Second number to add')

@tool
def add(data: AddInput) -> int:
    """Adds two numbers together."""
    return data.a + data.b
```

### Variant 2: Regular Parameters (Auto-Generated Schema)

Pass parameters directly like any normal function. TinyGent **auto-generates** a `TinyModel` schema from your type hints.

```python
from tinygent.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together."""
    return a * b
```

### When to Use Which?

| Variant | Best For | Pros | Cons |
|---------|----------|------|------|
| **TinyModel** | Complex tools, detailed schemas | Full control over field descriptions, validation rules | More boilerplate |
| **Regular params** | Simple utilities, quick prototyping | Minimal code, familiar syntax | Field descriptions derived from param names only |

Both variants:

1. **Schema validation** — arguments validated against Pydantic.
2. **LLM compatibility** — tools have OpenAI-compatible schemas for function calling.
3. **Optional auto-registration** — tools are globally available if you use `@register_tool`.

---

## Tool Types

TinyGent supports several kinds of tools:

| Type                        | Decorator(s)                            | Registered globally | Extra behavior |
|-----------------------------|-----------------------------------------|---------------------|----------------|
| **Local Tool**              | `@tool`                                 | ❌                  | Basic schema validation and execution |
| **Registered Tool**         | `@register_tool`                        | ✅                  | Auto-added to `GlobalToolCatalog`, accessible by name |
| **Reasoning Tool (local)**  | `@reasoning_tool`                       | ❌                  | Adds a `reasoning: str` field to the tool input, so the agent must state *why* it’s calling the tool |
| **Reasoning Tool (global)** | `@register_reasoning_tool`              | ✅                  | Same as above, but also auto-registered in the global catalog |
| **JIT Instruction Tool (local)**  | `@jit_tool`                     | ❌                  | Appends an `instruction` field to outputs so agents see on-the-fly guidance |
| **JIT Instruction Tool (global)** | `@register_jit_tool`            | ✅                  | Same as above, but also auto-registered in the global catalog |

---

### Reasoning Tools

Reasoning tools extend normal tools by requiring an additional field
`reasoning: str` in the input schema.  
When an LLM selects such a tool, it must also explain *why* it is calling it.  

This reasoning is captured and emitted through the **`on_tool_reasoning`** hook,  
allowing you to log or inspect the agent’s motivation behind each tool call.

---

## Tool Hooks

Whenever a tool is executed, the agent can emit the following [hooks](../agents/hooks/README.md):

* **`on_before_tool_call(tool_name, arguments)`** – Triggered before the tool runs.  
* **`on_after_tool_call(tool_name, arguments, result)`** – Triggered after the tool returns.  
* **`on_tool_reasoning(reasoning: str)`** – Triggered when a `ReasoningTool` provides reasoning.  

These hooks make it possible to monitor, debug, or extend tool usage.

---

## Decorators

### `@tool`

Creates a `Tool` instance but does **not** register it.

**Using TinyModel:**
```python
from tinygent.tools import tool

@tool
def local_add(data: AddInput) -> int:
    """Adds two numbers."""
    return data.a + data.b

print(local_add(AddInput(a=1, b=2)))  # works directly
```

**Using regular parameters:**
```python
from tinygent.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

print(multiply(a=3, b=4))  # -> 12
print(multiply({'a': 3, 'b': 4}))  # also works with dict
```

### `@register_tool`

Creates a `Tool` instance **and registers it** into the global registry.

**Using TinyModel:**
```python
from tinygent.tools import register_tool

@register_tool(use_cache=True)
def add(data: AddInput) -> int:
    """Adds two numbers together."""
    return data.a + data.b
```

**Using regular parameters:**
```python
from tinygent.tools import register_tool

@register_tool(use_cache=True)
def subtract(a: int, b: int) -> int:
    """Subtracts b from a."""
    return a - b

from tinygent.core.runtime.tool_catalog import GlobalToolCatalog

registry = GlobalToolCatalog().get_active_catalog()
add_tool = registry.get_tool('add')
print(add_tool(a=1, b=2))

```

### `@jit_tool`

Wraps an existing tool and augments every result with a lightweight instruction payload. This is useful when you want the LLM to receive inline guidance about how to interpret or execute the tool outcome.

```python
from tinygent.tools import jit_tool

@jit_tool(jit_instruction='Count from 1 to n, yielding each number.')
def count(data: CountInput):
    for i in range(1, data.n + 1):
        yield i

print(list(count(CountInput(n=3))))  # -> [1, 2, 3, {'instruction': 'Count from 1 to n, yielding each number.'}]
```

Use `@register_jit_tool` when you also want to publish the wrapped tool into the global catalog.
```

---

## Global vs Local Tools

You can decide whether a tool is registered globally or kept local.

* **Global tools** (`@register_tool`, `@register_reasoning_tool`)

  * Automatically stored in the `GlobalToolCatalog`
  * Discoverable by name from anywhere in your runtime
  * Great when you want tools to be available for LLM function calling across the system

* **Local tools** (`@tool`, `@reasoning_tool`)

  * Return a `Tool`\/`ReasoningTool` instance only
  * Not stored in the registry
  * You manage them explicitly (e.g., pass into `llm.generate_with_tools`)
  * Useful for ephemeral or experimental utilities

---

## Tool Features

Each decorated function becomes a `Tool`\/`ReasoningTool` instance that:

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
* **Not supported for generator tools** (`count`, `async_count`)
* Cache inspection and clearing:

```python
expensive_tool.cache_info()
expensive_tool.clear_cache()
```

⚠️ **Note:** For generator and async generator tools, `use_cache=True` is ignored and `cache_info()` always returns `None`.

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

# Access from global registry (registered tools)
from tinygent.core.runtime.tool_catalog import GlobalToolCatalog
registry = GlobalToolCatalog.get_registry()

print(registry.get_tool("greet")({"name": "TinyGent"}))

# Local-only tools (not in registry)
print(list(count(n=5)))
print(list(async_count({"n": 6})))
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
[1, 2, 3, 4, 5]
[1, 2, 3, 4, 5, 6]
```
