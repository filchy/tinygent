<div align="center" style="display: flex; align-items: center; justify-content: center; gap: 20px;">

  <picture style="display: flex; align-items: center;">
    <source media="(prefers-color-scheme: light)" srcset="docs/dark-avatar.png">
    <source media="(prefers-color-scheme: dark)" srcset="docs/light-avatar.png">
    <img alt="TinyCorp avatar" src="docs/light-avatar.png" style="height: 120px; width: auto;">
  </picture>

  <picture style="display: flex; align-items: center;">
    <source media="(prefers-color-scheme: light)" srcset="docs/dark-logo.png">
    <source media="(prefers-color-scheme: dark)" srcset="docs/light-logo.png">
    <img alt="TinyCorp logo" src="docs/light-logo.png" style="height: 120px; width: auto;">
  </picture>

</div>

___

Tinygent is a tiny agentic framework - lightweight, easy to use (hopefully), and efficient (also hopefully ;-0) library for building and deploying generative AI applications. It provides a simple interface for working with various models and tools, making it ideal for developers who want to quickly prototype and deploy AI solutions.

## Getting Started

### Prerequisites

Before you begin using tinygent, ensure that you meet the following software prerequisites.

- Install [Git](https://git-scm.com/)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Install From Source

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

## Examples (Quick Start)

1. Ensure you have set the `OPENAI_API_KEY` environment variable to allow the example to use OpenAI's API. An API key can be obtained from [`openai.com`](https://openai.com/).
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    ```

2. Run the examples using `uv`:
    ```bash
    uv run examples/agents/multi-step/main.py
    ```

3. Explore more examples below:

### Basics

1. [Tool Usage](examples/tool-usage)
2. [LLM Usage](examples/llm-usage)
3. [Function Calling](examples/function-calling)

### Memory

1. [Chat Buffer Memory](examples/memory/basic-chat-memory)
2. [Window Buffer Memory](examples/memory/buffer-window-chat-memory)
3. [Combined Memory](examples/memory/combined-memory)

### Tools

1. [Basic Tools](examples/tool-usage/main.py)
2. [Reasoning Tools](examples/tool-usage/main.py)
3. [JIT Tools](examples/tool-usage/main.py)

### Agents

1. [Hooks in Agents](examples/agents/hooks/)
2. [ReAct Agent](examples/agents/react/)
3. [Multi-Step Agent](examples/agents/multi-step/)
4. [Squad Agent](examples/agents/squad/)
5. [Modular Agentic Planner Agent](examples/agents/map/)

### Packages

1. [Brave Tools](packages/tiny_brave/)
2. [Tiny Chat](packages/tiny_chat)
3. [Tiny OpenAI](packages/tiny_openai)
4. [Tiny MistralAI](packages/tiny_mistralai)
5. [Tiny Gemini](packages/tiny_gemini)
6. [Tiny Anthropic](packages/tiny_anthropic)
7. [Tiny Graph](packages/tiny_graph)

## Linting & Formatting

To ensure code quality, formatting consistency, and type safety, run:

```bash
uv run fmt   # Format code Ruff
uv run lint  # Run Ruff linter and Mypy type checks
```
