<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/resources/images/logo-light.svg">
    <source media="(prefers-color-scheme: light)" srcset=".github/resources/images/logo-dark.svg">
    <img alt="TinyGent Logo" src=".github/resources/images/logo-dark.svg" width="80%">
  </picture>
</div>

<div align="center">
  <h3>✨ Tiny platform for amazing agents ✨</h3>

  <h4>
    🏠 <a href="https://github.com/filchy/tinygent">Homepage</a> | 📚 <a href="https://filchy.github.io/tinygent">Documentation</a> | 💡 <a href="https://filchy.github.io/tinygent/examples">Examples</a> | 🚀 <a href="https://filchy.github.io/tinygent/examples/quick-start/">Quick Start</a>
  </h4>
</div>

🤖 Tinygent is a tiny agentic framework - lightweight, easy to use (hopefully), and efficient (also hopefully ;-0) library for building and deploying generative AI applications. It provides a simple interface for working with various models and tools, making it ideal for developers who want to quickly prototype and deploy AI solutions.

## 🎯 Create an agent

```python
# uv sync --extra openai

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

print(agent.run('What is the weather like in Prague?'))
```

## 🚀 Getting Started

### 📋 Prerequisites

Before you begin using tinygent, ensure that you meet the following software prerequisites.

- Install [Git](https://git-scm.com/)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

### 💻 Install From Source

1. Clone the tinygent repository to your local machine.
    ```bash
    git clone git@github.com:filchy/tinygent.git tinygent
    cd tinygent
    ```

2. Create a Python environment.
    ```bash
    uv venv --seed .venv
    source .venv/bin/activate
    ```

3. Install the tinygent library.
    To install only the core tinygent library without any optional dependencies, run the following:
    ```bash
    uv sync
    ```

    To install the tinygent library along with all of the optional dependencies. Including developer tools (`--all-groups`), additional packages and all of the dependencies needed for profiling and plugins (`--all-extras`) in the source repository, run the following:
    ```bash
    uv sync --all-groups --all-extras
    ```

    > [!NOTE]
    > Not all packages are included in the default installation to keep the library lightweight. You can customize your installation by specifying the optional dependencies you need.

4. Install tinygent in editable mode (development mode), so that changes in the source code are immediately reflected:
    ```bash
    uv pip install -e .
    ```

## 🎬 See It In Action

<div align="center">
  <img alt="TinyChat Demo" src=".github/resources/gifs/demo.gif" width="100%">
</div>

## 🏗️ Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#e1f5ff', 'primaryTextColor': '#0d47a1', 'primaryBorderColor': '#42a5f5', 'lineColor': '#1976d2', 'secondaryColor': '#fff4e1', 'tertiaryColor': '#f0e1ff'}}}%%
flowchart LR
    User[User Code]:::userNode
    Factory[Factories]:::factoryNode
    Runtime[Runtime Registry]:::runtimeNode
    Components[Agents & Tools & Memory]:::componentNode
    Packages[Provider Packages]:::packageNode

    User a@--> Factory
    Factory b@--> Runtime
    Runtime c@--> Components
    Packages d@-.-> Runtime

    a@{animate: true}
    b@{animate: true}
    c@{animate: true}
    d@{animate: true}

    linkStyle 0 stroke-width:2px
    linkStyle 1 stroke-width:2px
    linkStyle 2 stroke-width:2px
    linkStyle 3 stroke-width:2px,stroke-dasharray: 5 5

    classDef userNode fill:#e1f5ff,stroke:#1976d2,stroke-width:2px,color:#0d47a1
    classDef factoryNode fill:#fff4e1,stroke:#f57c00,stroke-width:2px,color:#e65100
    classDef runtimeNode fill:#f0e1ff,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    classDef componentNode fill:#e1ffe1,stroke:#388e3c,stroke-width:2px,color:#1b5e20
    classDef packageNode fill:#ffe1e1,stroke:#d32f2f,stroke-width:2px,color:#b71c1c
```

Tinygent uses a registry-based plugin architecture: **Packages** register components into the **Runtime**. **Factories** query the Runtime to build **Components** for your code.

## 💡 Examples (Quick Start)

1. 🔑 Ensure you have set the `OPENAI_API_KEY` environment variable to allow the example to use OpenAI's API. An API key can be obtained from [`openai.com`](https://openai.com/).
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    ```

2. ▶️ Run the examples using `uv`:
    ```bash
    uv run examples/agents/multi-step/main.py
    ```

3. 🔍 Explore more examples below:

### 📚 Features & Examples

| Name | Type | Description |
|------|------|-------------|
| [Tool Usage](examples/tool-usage) | Basics | Basic tool creation and usage |
| [LLM Usage](examples/llm-usage) | Basics | Direct LLM interaction patterns |
| [Function Calling](examples/function-calling) | Basics | Function calling with LLMs |
| [Chat Buffer Memory](examples/memory/basic-chat-memory) | Memory | Store complete conversation history |
| [Summary Buffer Memory](examples/memory/buffer-summary-memory) | Memory | Summarize older messages to save tokens |
| [Window Buffer Memory](examples/memory/buffer-window-chat-memory) | Memory | Keep only recent N messages |
| [Combined Memory](examples/memory/combined-memory) | Memory | Combine multiple memory strategies |
| [Basic Tools](examples/tool-usage/main.py) | Tools | Simple tool definitions with `@tool` |
| [Reasoning Tools](examples/tool-usage/main.py) | Tools | Tools that provide reasoning traces |
| [JIT Tools](examples/tool-usage/main.py) | Tools | Just-in-time compiled tools |
| [Middlewares in Agents](examples/agents/middleware/) | Agents | Add custom processing layers to agents |
| [ReAct Agent](examples/agents/react/) | Agents | Reasoning and acting pattern |
| [Multi-Step Agent](examples/agents/multi-step/) | Agents | Break down complex tasks into steps |
| [Squad Agent](examples/agents/squad/) | Agents | Coordinate multiple specialized agents |
| [Modular Agentic Planner](examples/agents/map/) | Agents | Advanced planning with modular architecture |
| [Tiny OpenAI](packages/tiny_openai) | Package | OpenAI LLMs and embeddings |
| [Tiny Anthropic](packages/tiny_anthropic) | Package | Anthropic Claude LLMs |
| [Tiny MistralAI](packages/tiny_mistralai) | Package | Mistral AI LLMs |
| [Tiny Gemini](packages/tiny_gemini) | Package | Google Gemini LLMs |
| [Tiny VoyageAI](packages/tiny_voyageai) | Package | VoyageAI embedding models |
| [Brave Tools](packages/tiny_brave/) | Package | Brave search integration |
| [Tiny Chat](packages/tiny_chat) | Package | FastAPI-based chat interface |
| [Tiny Graph](packages/tiny_graph) | Package | Neo4j knowledge graph integration |

## ✅ Linting & Formatting

To ensure code quality, formatting consistency, and type safety, run:

```bash
uv run fmt   # Format code Ruff
uv run lint  # Run Ruff linter and Mypy type checks
```
