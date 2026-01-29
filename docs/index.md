## What is Tinygent?

Tinygent is a **lightweight, powerful agentic framework** for building generative AI applications. Unlike heavyweight frameworks that come with complexity overhead, Tinygent focuses on simplicity, flexibility, and developer experience.

Built with modern Python best practices, Tinygent provides:

- **Simple API** - Build agents in just a few lines of code
- **Multi-Provider Support** - OpenAI, Anthropic, Mistral, Gemini, and more
- **Flexible Tools** - Simple, reasoning, and JIT tool decorators
- **Smart Memory** - Buffer, summary, window, and combined memory types
- **Middleware** - Extensible agent behavior with middleware pattern
- **Async-First** - Built for high-performance async workflows
- **Modular Design** - Use only what you need

---

## Why Tinygent?

### **Start Simple, Scale Your Way**

Tinygent follows the principle of progressive disclosure - start simple, grow complex only when needed.

```python
from tinygent.tools import tool
from tinygent.core.factory import build_agent

@tool
def get_weather(location: str) -> str:
    """Get the current weather in a given location."""
    return f'The weather in {location} is sunny with a high of 75°F.'

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[get_weather],
)

result = agent.run('What is the weather like in Prague?')
print(result)
```

That's it! You've built a ReAct agent with tool-calling capabilities.

### **Provider Agnostic**

Switch between LLM providers with a single string:

```python
# OpenAI
agent = build_agent('react', llm='openai:gpt-4o-mini', tools=[...])

# Anthropic Claude
agent = build_agent('react', llm='anthropic:claude-3-5-sonnet', tools=[...])

# Mistral
agent = build_agent('react', llm='mistralai:mistral-large-latest', tools=[...])

# Google Gemini
agent = build_agent('react', llm='gemini:gemini-2.0-flash-exp', tools=[...])
```

### **Batteries Included, But Replaceable**

Tinygent comes with powerful built-in components, but everything is replaceable:

- **4 Agent Types**: ReAct, MultiStep, Squad, MAP (Modular Agentic Planner)
- **Multiple Memory Types**: Buffer, Summary, Window, Combined
- **Tool Decorators**: `@tool`, `@reasoning_tool`, `@jit_tool`
- **Middleware System**: Extensible hooks for customization
- **Optional Packages**: Brave search, Neo4j graphs, Chat UI, and more

---

## Core Principles

### **1. Config/Builder Pattern**

All components use a consistent configuration pattern:

```python
from tinygent.agents.react_agent import ReactAgentConfig

config = ReactAgentConfig(
    llm="openai:gpt-4o-mini",
    tools=[get_weather],
    max_iterations=5,
    temperature=0.7
)

agent = config.build()
```

### **2. Registry Pattern**

Components auto-register for global discovery:

```python
from tinygent.tools import register_tool

@register_tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

# Later, discover and use from anywhere
from tinygent.core.runtime.tool_catalog import GlobalToolCatalog

registry = GlobalToolCatalog().get_active_catalog()
search_tool = registry.get_tool('search')
```

### **3. Async-First**

All agent operations support async streaming:

```python
async def main():
    agent = build_agent('react', llm='openai:gpt-4o-mini', tools=[...])

    # Stream responses
    async for chunk in agent.run_stream('What is the weather?'):
        print(chunk, end='', flush=True)

# Synchronous wrapper available too
result = agent.run('What is the weather?')
```

### **4. Type Safety**

Built with Pydantic for runtime validation and excellent IDE support:

```python
from pydantic import Field
from tinygent.core.types import TinyModel

class WeatherInput(TinyModel):
    location: str = Field(..., description='The location to get weather for')
    units: str = Field('celsius', description='Temperature units')

@tool
def get_weather(data: WeatherInput) -> str:
    return f"Weather in {data.location}: 22°{data.units[0].upper()}"
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone git@github.com:filchy/tinygent.git
cd tinygent

# Create virtual environment
uv venv --seed .venv
source .venv/bin/activate

# Install core library
uv sync

# Install with OpenAI support
uv sync --extra openai
```

### Your First Agent

Create a file `my_agent.py`:

```python
from tinygent.tools import tool
from tinygent.core.factory import build_agent

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error: {e}"

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[calculator],
)

result = agent.run('What is 123 * 456?')
print(result)
```

Run it:

```bash
export OPENAI_API_KEY="your-api-key"
uv run my_agent.py
```

---

## What's Next?

<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting Started**

    ---

    Install Tinygent and build your first agent in minutes

    [:octicons-arrow-right-24: Installation Guide](getting-started.md)

-   :material-brain: **Core Concepts**

    ---

    Learn about agents, tools, LLMs, memory, and middleware

    [:octicons-arrow-right-24: Learn Concepts](concepts/agents.md)

-   :material-code-braces: **Examples**

    ---

    Explore practical examples and use cases

    [:octicons-arrow-right-24: View Examples](examples.md)

-   :material-book-open-variant: **API Reference**

    ---

    Detailed API documentation for all components

    [:octicons-arrow-right-24: API Docs](api-reference.md)

</div>

---

## Community & Support

- **GitHub**: [github.com/filchy/tinygent](https://github.com/filchy/tinygent)
- **Issues**: [Report bugs or request features](https://github.com/filchy/tinygent/issues)
- **Examples**: Check the `examples/` directory for more code samples

---

## License

Tinygent is open source software. Check the repository for license details.
