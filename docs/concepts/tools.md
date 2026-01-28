# Tools

Tools are functions that agents can call to interact with the world. Tinygent makes it easy to turn any Python function into an agent-compatible tool.

---

## What is a Tool?

A tool is a **Python function** that:

1. Has a clear purpose (described in docstring)
2. Has typed parameters (for schema generation)
3. Returns a value (for agent observation)
4. Is decorated with `@tool`, `@register_tool`, `@reasoning_tool`, or `@jit_tool`

**Example:**

```python
from tinygent.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the current weather in a given location.

    Args:
        location: The city or location to get weather for

    Returns:
        A string describing the weather
    """
    return f'The weather in {location} is sunny with a high of 75°F.'
```

When you pass this tool to an agent, the LLM can:

- See the function name: `get_weather`
- See the description: `"Get the current weather..."`
- See the parameters: `location: str`
- Call it when needed: `get_weather(location="Prague")`
- Use the result: `"The weather in Prague is sunny..."`

---

## Tool Decorators

Tinygent provides 4 tool decorators for different use cases:

### 1. `@tool` - Simple Tools

**Best for**: Quick local tools, no global registration needed

```python
from tinygent.tools import tool

@tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression.

    Args:
        expression: A valid Python math expression

    Returns:
        The result of the calculation
    """
    return eval(expression)

# Use directly in agent
agent = build_agent('react', llm='openai:gpt-4o-mini', tools=[calculator])
```

**Features:**

- Lightweight, no registration
- Great for simple use cases
- Perfect for inline definitions

---

### 2. `@register_tool` - Globally Registered Tools

**Best for**: Reusable tools, CLI usage, multi-agent systems

```python
from tinygent.tools import register_tool

@register_tool(use_cache=True)
def search_web(query: str) -> str:
    """Search the web for information."""
    # Call web search API
    return f"Results for: {query}"

# Discover from global registry
from tinygent.core.runtime.tool_catalog import GlobalToolCatalog

registry = GlobalToolCatalog().get_active_catalog()
search = registry.get_tool('search_web')
```

**Features:**

- **Global discovery**: Access from anywhere
- **Caching**: Optional result caching with `use_cache=True`
- **CLI support**: Usable in `tiny` CLI terminal
- **Reusability**: Share across multiple agents

**Caching Example:**

```python
@register_tool(use_cache=True)
def expensive_api_call(query: str) -> str:
    """Call an expensive API."""
    import time
    time.sleep(2)  # Simulate slow API
    return f"Result for {query}"

# First call: Takes 2 seconds
result1 = expensive_api_call(query="test")

# Second call with same args: Instant (cached)
result2 = expensive_api_call(query="test")

# Check cache stats
print(expensive_api_call.cache_info())
# CacheInfo(hits=1, misses=1, maxsize=128, currsize=1)

# Clear cache
expensive_api_call.clear_cache()
```

---

### 3. `@reasoning_tool` - Tools with Reasoning

**Best for**: Complex operations that benefit from explaining "why"

Reasoning tools require the agent to provide a rationale before calling:

```python
from tinygent.tools import register_reasoning_tool

@register_reasoning_tool(
    reasoning_prompt='Explain why you are performing this search.'
)
def search_database(query: str) -> str:
    """Search the internal database."""
    return f"Database results for: {query}"
```

**Agent interaction:**

```
Agent: I need to search the database for user information.
Reasoning: I'm searching for "John Doe" because the user asked about their account status.
Action: search_database(query="John Doe")
Observation: Found user record for John Doe
```

**When to use:**

- High-cost operations (API calls, computations)
- Actions that need justification (delete, modify)
- Debugging agent decision-making

---

### 4. `@jit_tool` - Just-In-Time Code Generation

**Best for**: Dynamic operations, code generation, flexible workflows

JIT tools generate and execute code at runtime based on agent instructions:

```python
from tinygent.tools import jit_tool

@jit_tool(jit_instruction='Generate code to count from 1 to n, yielding each number.')
def count(n: int):
    """Count from 1 to n, yielding each number."""
    for i in range(1, n + 1):
        yield i

# Agent can dynamically modify behavior
result = list(count(n=5))  # [1, 2, 3, 4, 5]
```

**When to use:**

- Dynamic code generation
- Flexible operations
- When tool behavior needs runtime customization

---

## Tool Schemas

Tools can accept two input styles:

### Style 1: Pydantic Models

**Best for**: Complex inputs, validation, documentation

```python
from pydantic import Field
from tinygent.core.types import TinyModel

class WeatherInput(TinyModel):
    location: str = Field(..., description='The city or location')
    units: str = Field('celsius', description='Temperature units: celsius or fahrenheit')
    include_forecast: bool = Field(False, description='Include 5-day forecast')

@register_tool
def get_weather(data: WeatherInput) -> str:
    """Get detailed weather information."""
    forecast = ' + 5-day forecast' if data.include_forecast else ''
    return f"Weather in {data.location}: 22°{data.units[0].upper()}{forecast}"
```

**Benefits:**

- Runtime validation (Pydantic)
- Rich field descriptions
- Default values
- Type safety

### Style 2: Regular Parameters

**Best for**: Simple inputs, quick tools

```python
@register_tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

# Call with kwargs
result = multiply(a=3, b=4)  # 12

# Or with dict
result = multiply({'a': 5, 'b': 6})  # 30
```

**Benefits:**

- Less boilerplate
- Quick to write
- Familiar Python syntax

---

## Async Tools

Tools can be async for I/O operations:

```python
import httpx

@register_tool
async def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text

# Agents automatically handle async tools
agent = build_agent('react', llm='openai:gpt-4o-mini', tools=[fetch_url])
result = agent.run('Fetch https://example.com')
```

---

## Generator Tools

Tools can yield results for streaming:

```python
@tool
async def stream_data(query: str):
    """Stream data from a source."""
    for i in range(5):
        await asyncio.sleep(0.5)
        yield f"Chunk {i}: {query}"

# Agent receives results as they arrive
result = list(stream_data(query="test"))
# ['Chunk 0: test', 'Chunk 1: test', ...]
```

---

## Error Handling

Tools should raise descriptive errors:

```python
@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Agent receives error and tries alternative approach
agent = build_agent('react', llm='openai:gpt-4o-mini', tools=[divide])
result = agent.run('What is 10 divided by 0?')
# Agent: "Division by zero is undefined in mathematics."
```

---

## Tool Composition

Combine tools for complex workflows:

```python
@register_tool
def search_products(category: str) -> list[str]:
    """Search for products in a category."""
    return ['Product A', 'Product B', 'Product C']

@register_tool
def get_product_details(product_name: str) -> dict:
    """Get detailed information about a product."""
    return {
        'name': product_name,
        'price': 99.99,
        'in_stock': True
    }

# Agent chains tools
agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[search_products, get_product_details]
)

result = agent.run('Find electronics and tell me about Product A')
# Agent calls: search_products('electronics') → get_product_details('Product A')
```

---

## Best Practices

### 1. Clear Docstrings

```python
# Bad
@tool
def process(data: str) -> str:
    """Process data."""  # Too vague
    return data.upper()

# Good
@tool
def uppercase_text(text: str) -> str:
    """Convert text to uppercase.

    Args:
        text: The text to convert

    Returns:
        The text in uppercase

    Example:
        uppercase_text("hello") -> "HELLO"
    """
    return text.upper()
```

### 2. Type Hints

```python
# Bad
@tool
def add(a, b):  # No types
    return a + b

# Good
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b
```

### 3. Single Responsibility

```python
# Bad - Does too much
@tool
def fetch_and_analyze_and_summarize(url: str) -> str:
    """Fetch URL, analyze content, and summarize."""
    content = fetch(url)
    analyzed = analyze(content)
    return summarize(analyzed)

# Good - Split into focused tools
@tool
def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    return fetch(url)

@tool
def analyze_text(text: str) -> dict:
    """Analyze text content."""
    return analyze(text)

@tool
def summarize_analysis(analysis: dict) -> str:
    """Create summary from analysis."""
    return summarize(analysis)
```

### 4. Validation

```python
from pydantic import Field, field_validator

class SearchInput(TinyModel):
    query: str = Field(..., min_length=1, max_length=200)
    max_results: int = Field(10, ge=1, le=100)

    @field_validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

@register_tool
def search(data: SearchInput) -> str:
    """Search with validation."""
    return f"Searching for: {data.query}"
```

---

## Tool Registry

Access registered tools globally:

```python
from tinygent.core.runtime.tool_catalog import GlobalToolCatalog

# Get active catalog
catalog = GlobalToolCatalog().get_active_catalog()

# List all tools
all_tools = catalog.list_tools()
print(all_tools)  # ['get_weather', 'search_web', 'calculator', ...]

# Get specific tool
weather_tool = catalog.get_tool('get_weather')

# Call it
result = weather_tool(location='Prague')
```

---

## Advanced: Hidden Tools

Mark tools as hidden for internal use:

```python
@register_tool(hidden=True)
def internal_helper(data: str) -> str:
    """Internal tool, not exposed to agents."""
    return process_internally(data)

# Not visible in default tool lists
catalog = GlobalToolCatalog().get_active_catalog()
visible_tools = catalog.list_tools()  # Doesn't include 'internal_helper'

# But still accessible if you know the name
helper = catalog.get_tool('internal_helper')
```

---

## Next Steps

- **[Agents](agents.md)**: Use tools with agents
- **[Custom Tools Guide](../guides/custom-tools.md)**: Build advanced tools
- **[Examples](../examples.md)**: See tool usage examples

---

## Examples

Check out:

- `examples/tool-usage/main.py` - All tool decorator types
- `examples/function-calling/main.py` - Function calling patterns
- `packages/tiny_brave/` - Real-world tool: Brave search
