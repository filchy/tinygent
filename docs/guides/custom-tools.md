# Building Custom Tools

A comprehensive guide to creating powerful custom tools for your agents.

---

## Quick Start

```python
from tinygent.tools import tool

@tool
def hello_world(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"

# Use it
from tinygent.core.factory import build_agent

agent = build_agent('react', llm='openai:gpt-4o-mini', tools=[hello_world])
result = agent.run('Say hello to Alice')
```

---

## Tool Anatomy

Every tool needs three things:

1. **Decorator**: `@tool`, `@register_tool`, `@reasoning_tool`, or `@jit_tool`
2. **Type hints**: For automatic schema generation
3. **Docstring**: Describes what the tool does

```python
from tinygent.tools import tool

@tool  # 1. Decorator
def search_database(query: str, limit: int = 10) -> list[dict]:  # 2. Type hints
    """Search the database for records matching the query.  # 3. Docstring

    Args:
        query: The search term to look for
        limit: Maximum number of results to return

    Returns:
        List of matching records
    """
    # Implementation
    results = database.search(query, limit=limit)
    return results
```

---

## Simple Tools

### Basic Function Tool

```python
@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
```

### With Default Values

```python
@tool
def greet(name: str, greeting: str = "Hello") -> str:
    """Greet someone with a custom greeting."""
    return f"{greeting}, {name}!"

# Agent can call:
# greet(name="Alice") → "Hello, Alice!"
# greet(name="Bob", greeting="Hi") → "Hi, Bob!"
```

### With Optional Parameters

```python
from typing import Optional

@tool
def send_email(to: str, subject: str, body: str, cc: Optional[str] = None) -> str:
    """Send an email.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body content
        cc: Optional CC recipient
    """
    message = f"Sending email to {to}"
    if cc:
        message += f" (CC: {cc})"
    return message
```

---

## Pydantic Model Tools

For complex validation and documentation:

```python
from pydantic import Field, field_validator, EmailStr
from tinygent.core.types import TinyModel
from tinygent.tools import register_tool

class EmailInput(TinyModel):
    to: EmailStr = Field(..., description='Recipient email address')
    subject: str = Field(..., min_length=1, max_length=100)
    body: str = Field(..., min_length=1, description='Email body content')
    priority: str = Field('normal', description='Priority: low, normal, high')

    @field_validator('priority')
    def validate_priority(cls, v):
        allowed = ['low', 'normal', 'high']
        if v not in allowed:
            raise ValueError(f'Priority must be one of: {allowed}')
        return v

@register_tool
def send_email(data: EmailInput) -> str:
    """Send an email with validation."""
    return f"Sent {data.priority}-priority email to {data.to}"
```

---

## Async Tools

For I/O-bound operations:

```python
import httpx

@register_tool
async def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text

@register_tool
async def query_database(query: str) -> list[dict]:
    """Query async database."""
    async with async_db.connect() as conn:
        results = await conn.fetch(query)
        return [dict(row) for row in results]
```

---

## Generator Tools

For streaming results:

```python
@tool
async def stream_search_results(query: str) -> str:
    """Stream search results one at a time."""
    for i in range(5):
        await asyncio.sleep(0.5)
        yield f"Result {i+1}: {query}"

# Agent receives:
# "Result 1: my query"
# "Result 2: my query"
# ...
```

---

## Error Handling

### Validation Errors

```python
@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero. Please use a non-zero divisor.")
    return a / b

# Agent will see: "Error: Cannot divide by zero. Please use a non-zero divisor."
```

### API Errors

```python
import httpx

@tool
async def call_api(endpoint: str) -> dict:
    """Call external API."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, timeout=5.0)
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        raise Exception("API request timed out after 5 seconds")
    except httpx.HTTPError as e:
        raise Exception(f"API error: {e}")
```

---

## Caching Tools

Speed up repeated calls:

```python
@register_tool(use_cache=True)
def expensive_computation(input_data: str) -> str:
    """Perform expensive computation (cached)."""
    import time
    time.sleep(2)  # Simulate slow operation
    return f"Computed: {input_data.upper()}"

# First call: Takes 2 seconds
result = expensive_computation(input_data="test")

# Second call with same input: Instant (from cache)
result = expensive_computation(input_data="test")

# Different input: Takes 2 seconds again
result = expensive_computation(input_data="other")

# Cache stats
print(expensive_computation.cache_info())
# CacheInfo(hits=1, misses=2, maxsize=128, currsize=2)

# Clear cache
expensive_computation.clear_cache()
```

---

## Reasoning Tools

Require the agent to explain its reasoning:

```python
from tinygent.tools import register_reasoning_tool

@register_reasoning_tool(
    reasoning_prompt='Explain why you need to delete this record.'
)
def delete_record(record_id: int) -> str:
    """Delete a record from the database.

    This is a destructive operation and requires reasoning.

    Args:
        record_id: ID of the record to delete
    """
    # Agent must provide reasoning before calling
    return f"Deleted record {record_id}"

# Agent interaction:
# Thought: I need to delete record 123
# Reasoning: The user requested to remove their old account
# Action: delete_record(record_id=123)
# Observation: Deleted record 123
```

---

## JIT Tools

Generate code at runtime:

```python
from tinygent.tools import jit_tool

@jit_tool(
    jit_instruction='Generate code to process data according to user requirements.'
)
def dynamic_processor(data: str):
    """Dynamically process data based on agent-generated code."""
    # Agent generates and executes code
    yield from process_data(data)

# Agent can adapt behavior at runtime
```

---

## Real-World Examples

### 1. Web Scraper

```python
from bs4 import BeautifulSoup
import httpx

class ScraperInput(TinyModel):
    url: str = Field(..., description='URL to scrape')
    selector: str = Field('p', description='CSS selector for content')

@register_tool
async def scrape_webpage(data: ScraperInput) -> str:
    """Scrape content from a webpage.

    Args:
        url: The webpage URL
        selector: CSS selector for elements to extract

    Returns:
        Extracted text content
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(data.url, timeout=10.0)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        elements = soup.select(data.selector)
        content = '\n'.join(el.get_text(strip=True) for el in elements)

        return content or "No content found with that selector"

    except Exception as e:
        raise Exception(f"Scraping failed: {e}")
```

### 2. File Operations

```python
from pathlib import Path

@register_tool
def read_file(filepath: str) -> str:
    """Read contents of a file.

    Args:
        filepath: Path to the file to read

    Returns:
        File contents as string
    """
    try:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        return path.read_text(encoding='utf-8')

    except Exception as e:
        raise Exception(f"Error reading file: {e}")

@register_tool
def write_file(filepath: str, content: str) -> str:
    """Write content to a file.

    Args:
        filepath: Path to the file to write
        content: Content to write to the file

    Returns:
        Success message
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')

        return f"Successfully wrote {len(content)} characters to {filepath}"

    except Exception as e:
        raise Exception(f"Error writing file: {e}")
```

### 3. Database Query

```python
import sqlite3
from typing import List, Dict

class QueryInput(TinyModel):
    sql: str = Field(..., description='SQL query to execute')
    database: str = Field('app.db', description='Database file path')

@register_tool
def query_database(data: QueryInput) -> List[Dict]:
    """Execute a SQL query and return results.

    Args:
        sql: The SQL query to execute (SELECT only)
        database: Path to SQLite database file

    Returns:
        List of result rows as dictionaries
    """
    # Security: Only allow SELECT queries
    if not data.sql.strip().upper().startswith('SELECT'):
        raise ValueError("Only SELECT queries are allowed")

    try:
        conn = sqlite3.connect(data.database)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(data.sql)
        results = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return results

    except Exception as e:
        raise Exception(f"Database error: {e}")
```

### 4. API Client

```python
class APICallInput(TinyModel):
    endpoint: str = Field(..., description='API endpoint path')
    method: str = Field('GET', description='HTTP method')
    params: dict = Field(default_factory=dict, description='Query parameters')

@register_tool
async def call_rest_api(data: APICallInput) -> dict:
    """Call a REST API endpoint.

    Args:
        endpoint: API endpoint path
        method: HTTP method (GET, POST, etc.)
        params: Query parameters or JSON body

    Returns:
        API response as dictionary
    """
    base_url = "https://api.example.com"
    url = f"{base_url}/{data.endpoint.lstrip('/')}"

    try:
        async with httpx.AsyncClient() as client:
            if data.method.upper() == 'GET':
                response = await client.get(url, params=data.params)
            elif data.method.upper() == 'POST':
                response = await client.post(url, json=data.params)
            else:
                raise ValueError(f"Unsupported method: {data.method}")

            response.raise_for_status()
            return response.json()

    except Exception as e:
        raise Exception(f"API call failed: {e}")
```

---

## Best Practices

### 1. Clear Descriptions

```python
# Bad
@tool
def process(data: str) -> str:
    """Process data."""  # Too vague
    return data.upper()

# Good
@tool
def convert_to_uppercase(text: str) -> str:
    """Convert text to uppercase letters.

    Args:
        text: The text to convert

    Returns:
        The text in uppercase

    Example:
        convert_to_uppercase("hello") → "HELLO"
    """
    return text.upper()
```

### 2. Validate Inputs

```python
from pydantic import Field, field_validator

class SearchInput(TinyModel):
    query: str = Field(..., min_length=1, max_length=200)
    limit: int = Field(10, ge=1, le=100)

    @field_validator('query')
    def clean_query(cls, v):
        # Remove extra whitespace
        return ' '.join(v.split())

@register_tool
def search(data: SearchInput) -> str:
    """Search with validated inputs."""
    return f"Searching for: {data.query}"
```

### 3. Return Structured Data

```python
from typing import List, Dict

@tool
def search_products(category: str) -> List[Dict[str, any]]:
    """Search for products.

    Returns structured data for easy parsing.
    """
    return [
        {'id': 1, 'name': 'Product A', 'price': 99.99},
        {'id': 2, 'name': 'Product B', 'price': 149.99},
    ]
```

### 4. Handle Edge Cases

```python
@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers with edge case handling."""
    if b == 0:
        raise ValueError("Cannot divide by zero")

    if abs(b) < 1e-10:
        raise ValueError("Divisor too close to zero")

    result = a / b

    if not math.isfinite(result):
        raise ValueError("Result is infinite or NaN")

    return result
```

### 5. Use Caching Wisely

```python
# Cache deterministic, expensive operations
@register_tool(use_cache=True)
def calculate_fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number (cached)."""
    # Expensive, but deterministic
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Don't cache non-deterministic operations
@register_tool(use_cache=False)  # Explicitly disable
def get_current_time() -> str:
    """Get current timestamp (not cached)."""
    # Non-deterministic - changes every call
    return datetime.now().isoformat()
```

---

## Testing Tools

### Unit Tests

```python
def test_add_tool():
    """Test the add tool."""
    result = add(a=2, b=3)
    assert result == 5

def test_add_with_dict():
    """Test calling with dict input."""
    result = add({'a': 5, 'b': 7})
    assert result == 12

def test_validation_error():
    """Test validation errors."""
    with pytest.raises(ValueError):
        divide(a=10, b=0)
```

### Integration Tests

```python
async def test_tool_with_agent():
    """Test tool integration with agent."""
    agent = build_agent(
        'react',
        llm='openai:gpt-4o-mini',
        tools=[add, multiply]
    )

    result = agent.run('What is 5 + 3 multiplied by 2?')
    assert '16' in result
```

---

## Next Steps

- **[Building Agents](building-agents.md)**: Use your tools with agents
- **[Tool Concepts](../concepts/tools.md)**: Deep dive into tools
- **[Examples](../examples.md)**: More tool examples

---

## Further Reading

- **Tool Implementation**: See `tinygent/tools/` for tool decorators
- **Tool Catalog**: See `tinygent/core/runtime/tool_catalog.py` for registry
- **Real Tools**: Check `packages/tiny_brave/` for a production tool example
