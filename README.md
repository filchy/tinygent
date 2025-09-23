<div align="center">
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="docs/dark-logo.png">
        <source media="(prefers-color-scheme: dark)" srcset="docs/light-logo.png">
        <img alt="TinyCorp logo" src="docs/light-logo.png" width="50%" height="50%">
    </picture>
</div>

___

Tinygent is a lightweight, easy-to-use (hopefully), and efficient (also hopefully ;-0) library for building and deploying generative AI applications. It provides a simple interface for working with various models and tools, making it ideal for developers who want to quickly prototype and deploy AI solutions.

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

    To install the tinygent library along with all of the optional dependencies. Including developer tools (`--all-groups`) and all of the dependencies needed for profiling and plugins (`--all-extras`) in the source repository, run the following:
    ```bash
    uv sync --all-groups --all-extras
    ```

4. Install tinygent in editable mode (development mode), so that changes in the source code are immediately reflected:
    ```bash
    uv pip install -e .
    ```

## Examples

### Basics

1. [Tool Usage](examples/tool-usage)
2. [LLM Usage](examples/llm-usage)
3. [Function Calling](examples/function-calling)

### Memory

1. [Chat Buffer Memory](examples/memory/basic-chat-memory)
2. [Window Buffer Memory](examples/memory/buffer-window-chat-memory)

### Agents

1. [ReAct Agent](examples/agents/react)

## Linting & Formatting

To ensure code quality, formatting consistency, and type safety, run:

```bash
uv run fmt   # Format code using Black and Ruff
uv run lint  # Run Ruff linter and Mypy type checks
```
