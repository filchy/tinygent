# Getting Started

This guide will help you install Tinygent and create your first agent.

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed
- **[Git](https://git-scm.com/)** for cloning the repository
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** - Modern Python package manager

### Installing uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Via pip
pip install uv
```

---

## Installation

### From Source

1. **Clone the repository**

```bash
git clone git@github.com:filchy/tinygent.git
cd tinygent
```

2. **Create a virtual environment**

```bash
uv venv --seed .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Tinygent**

**Option A: Core only** (minimal installation)

```bash
uv sync
```

**Option B: With specific providers**

```bash
# OpenAI
uv sync --extra openai

# Anthropic
uv sync --extra anthropic

# Multiple providers
uv sync --extra openai --extra anthropic --extra mistralai
```

**Option C: Everything** (including dev tools and all packages)

```bash
uv sync --all-groups --all-extras
```

4. **Install in editable mode** (for development)

```bash
uv pip install -e .
```

---

## Configuration

### API Keys

Tinygent uses environment variables for API keys. Set them before running your code:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Mistral AI
export MISTRAL_API_KEY="..."

# Google Gemini
export GEMINI_API_KEY="..."

# VoyageAI (embeddings)
export VOYAGEAI_API_KEY="..."

# Brave Search
export BRAVE_API_KEY="..."
```

**Pro Tip**: Create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Then load it in your code:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Your First Agent

Let's build a simple weather assistant.

### Step 1: Create a Tool

Tools are functions that agents can call. Use the `@tool` decorator:

```python
from tinygent.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the current weather in a given location.

    Args:
        location: The city or location to get weather for

    Returns:
        A string describing the current weather
    """
    # In a real app, you'd call a weather API
    return f'The weather in {location} is sunny with a high of 75°F.'
```

### Step 2: Build an Agent

Use the factory function to create a ReAct agent:

```python
from tinygent.core.factory import build_agent

agent = build_agent(
    'react',                    # Agent type
    llm='openai:gpt-4o-mini',  # LLM provider:model
    tools=[get_weather],        # List of tools
    max_iterations=5,           # Max reasoning loops
)
```

### Step 3: Run the Agent

```python
# Synchronous
result = agent.run('What is the weather like in Prague?')
print(result)

# Asynchronous with streaming
import asyncio

async def main():
    async for chunk in agent.run_stream('What is the weather in Prague?'):
        print(chunk, end='', flush=True)

asyncio.run(main())
```

### Complete Example

```python
# weather_agent.py
from tinygent.tools import tool
from tinygent.core.factory import build_agent

@tool
def get_weather(location: str) -> str:
    """Get the current weather in a given location."""
    return f'The weather in {location} is sunny with a high of 75°F.'

@tool
def get_forecast(location: str, days: int = 3) -> str:
    """Get the weather forecast for the next few days."""
    return f'{days}-day forecast for {location}: Sunny, then partly cloudy.'

def main():
    agent = build_agent(
        'react',
        llm='openai:gpt-4o-mini',
        tools=[get_weather, get_forecast],
    )

    result = agent.run(
        'What is the weather in Prague? Also give me a 5-day forecast.'
    )
    print(result)

if __name__ == '__main__':
    main()
```

Run it:

```bash
export OPENAI_API_KEY="your-key"
uv run weather_agent.py
```

---

## Understanding the Output

When you run the agent, you'll see the ReAct reasoning cycle:

1. **Thought**: Agent reasons about what to do
2. **Action**: Agent decides to call a tool
3. **Observation**: Tool returns a result
4. **Repeat**: Until agent has enough information
5. **Final Answer**: Agent provides the answer to the user

---

## Next Steps

Now that you have a working agent, explore more concepts:

- **[Agents](concepts/agents.md)**: Learn about different agent types (ReAct, MultiStep, Squad, MAP)
- **[Tools](concepts/tools.md)**: Discover `@reasoning_tool` and `@jit_tool`
- **[Memory](concepts/memory.md)**: Add conversation memory to your agents
- **[LLMs](concepts/llms.md)**: Use different LLM providers
- **[Middleware](concepts/middleware.md)**: Customize agent behavior with hooks

---

## Running Examples

Tinygent includes many examples in the `examples/` directory:

```bash
# ReAct agent
uv run examples/agents/react/main.py

# Multi-step agent
uv run examples/agents/multi-step/main.py

# Memory examples
uv run examples/memory/basic-chat-memory/main.py

# Tool usage
uv run examples/tool-usage/main.py
```

Explore the examples to see advanced patterns and use cases.

---

## Troubleshooting

### Import Errors

If you get import errors, ensure you've installed the package:

```bash
uv pip install -e .
```

### Missing Dependencies

Install the provider you need:

```bash
uv sync --extra openai
```

### API Key Issues

Verify your environment variables are set:

```bash
echo $OPENAI_API_KEY
```

---

## Development Setup

For contributors and advanced users:

```bash
# Install with all dev dependencies
uv sync --all-groups --all-extras

# Format code
uv run fmt

# Run linter and type checker
uv run lint

# Run tests (if available)
pytest
```

---

## What's Next?

<div class="grid cards" markdown>

-   :material-brain: **Learn Core Concepts**

    ---

    Understand agents, tools, and memory

    [:octicons-arrow-right-24: Core Concepts](concepts/agents.md)

-   :material-code-json: **Build Custom Tools**

    ---

    Create powerful custom tools for your agents

    [:octicons-arrow-right-24: Tool Guide](guides/custom-tools.md)

-   :material-robot: **Advanced Agents**

    ---

    Build multi-agent systems and complex workflows

    [:octicons-arrow-right-24: Agent Guide](guides/building-agents.md)

</div>
