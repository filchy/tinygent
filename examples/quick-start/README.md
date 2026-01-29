# Quick Start

A minimal example demonstrating how to create a ReAct agent with a custom tool using Tinygent.

## What This Example Does

This example creates a simple agent that can answer questions about the weather by using a custom `get_weather` tool. It demonstrates:

- Defining a custom tool with the `@tool` decorator
- Building a ReAct agent using the factory pattern
- Running the agent with a simple query

## Prerequisites

- Python 3.12
- OpenAI API key
- `uv` package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

## Setup

1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   ```

2. Install dependencies:
   ```bash
   uv sync --extra openai
   ```

## Run the Example

```bash
uv run examples/quick-start/main.py
```

## Expected Output

The agent will:
1. Receive the query "What is the weather like in Prague?"
2. Reason about which tool to use
3. Call the `get_weather` tool with location="Prague"
4. Return a natural language response based on the tool output

Example output:
```
The weather in Prague is sunny with a high of 75°F.
```

## Code Breakdown

```python
@tool
def get_weather(location: str) -> str:
    """Get the current weather in a given location."""
    return f'The weather in {location} is sunny with a high of 75°F.'
```
The `@tool` decorator registers a Python function as a tool that the agent can use. The docstring is used to help the agent understand when to use this tool.

```python
agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[get_weather],
)
```
`build_agent()` creates a ReAct agent using:
- `'react'` - The agent type (ReAct pattern: Reasoning + Acting)
- `llm='openai:gpt-4o-mini'` - The LLM to use (format: `provider:model`)
- `tools=[get_weather]` - List of tools available to the agent

```python
print(agent.run('What is the weather like in Prague?'))
```
`agent.run()` executes the agent with the given query and returns the final response.

## Next Steps

Explore more advanced examples:

- [Multi-Step Agent](../agents/multi-step/) - Agent that breaks down complex tasks
- [Tool Usage](../tool-usage/) - Advanced tool patterns (reasoning, JIT)
- [Memory Examples](../memory/) - Adding conversation memory to agents
- [Squad Agent](../agents/squad/) - Coordinating multiple agents

## Documentation

Visit the [Tinygent documentation](https://filchy.github.io/tinygent) for comprehensive guides and API references.
