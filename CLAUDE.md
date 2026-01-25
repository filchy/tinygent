# CLAUDE.md - Tinygent Development Guide

## Project Overview

Tinygent is a lightweight agentic framework for building and deploying generative AI applications. It provides a simple interface for working with various LLM providers (OpenAI, Anthropic, Mistral, Gemini) and tools.

## Architecture

### Main Modules

| Module | Path | Purpose |
|--------|------|---------|
| **agents** | `tinygent/agents/` | Agent implementations (ReAct, MultiStep, Squad, MAP) |
| **core** | `tinygent/core/` | Data models, factories, runtime registry, telemetry, prompts |
| **tools** | `tinygent/tools/` | Tool decorators (`@simple`, `@reasoning`, `@jit`) |
| **memory** | `tinygent/memory/` | Conversation memory (buffer, summary, window, combined) |
| **cli** | `tinygent/cli/` | `tiny` CLI command |
| **llms** | `tinygent/llms/` | LLM utilities |

### Runtime (`tinygent/core/runtime/`)

The runtime module provides global registries that store all registered components. When you use `@register_*` decorators (see Key Patterns below), components are stored in these registries and can later be retrieved via factory functions.

| Component | Purpose |
|-----------|---------|
| `global_registry.py` | Central registry for all component types (agents, LLMs, embedders, cross-encoders, memories, tools). Decorators like `@register_agent`, `@register_llm`, etc. populate this registry. |
| `tool_catalog.py` | Runtime tool registry with caching support and hidden tools. Used by `@register_tool` decorator. |
| `middleware_catalog.py` | Registry for agent middleware factories. Used by `@register_middleware` decorator. |
| `executors.py` | Execution utilities |

**Flow:** `@register_* decorator` → stores in `global_registry` / `tool_catalog` / `middleware_catalog` → retrieved via `build_*()` factory functions

### Packages (packages/)

Provider-specific implementations registered via entry points:

- `tiny_openai` - OpenAI LLM & embeddings
- `tiny_anthropic` - Anthropic Claude
- `tiny_mistralai` - Mistral AI
- `tiny_gemini` - Google Gemini
- `tiny_voyageai` - VoyageAI embeddings
- `tiny_brave` - Brave search
- `tiny_chat` - Chat UI (FastAPI)
- `tiny_graph` - Knowledge graph (Neo4j)

## Development Setup

**Always use `uv` as the package manager.**

```bash
# Create virtual environment
uv venv --seed .venv
source .venv/bin/activate

# Install core dependencies
uv sync

# Install all dependencies (dev + all optional packages)
uv sync --all-groups --all-extras

# Install in editable mode
uv pip install -e .

# Run examples
uv run examples/agents/multi-step/main.py

# Format code
uv run fmt

# Lint and type check
uv run lint
```

## Workflow Rules

- **NEVER commit without explicit user approval** - always show the diff and wait for approval
- **NEVER amend commits** - always create a new commit instead
- **ALWAYS run before committing** - `uv run fmt && uv run lint`

## Style Notes

- 4-space indentation, 89 char line limit
- Force single-line imports
- NEVER use emojis

## Key Patterns

### Config/Builder Pattern

All components use a `*Config` class with a `.build()` factory method:

```python
config = ReactAgentConfig(llm="openai:gpt-4o-mini", tools=[...])
agent = config.build()
```

### LLM String Syntax

Format: `"provider:model"` (e.g., `"openai:gpt-4o-mini"`, `"anthropic:claude-3-5-sonnet"`)

### Registry Pattern

Components auto-register via decorators:

- `@register_agent` - Agents
- `@register_tool` - Tools (simple)
- `@register_reasoning_tool` - Reasoning tools
- `@register_jit_tool` - JIT tools
- `@register_memory` - Memory types
- `@register_middleware` - Middleware
- `@register_llm` - LLM providers
- `@register_embedder` - Embedders
- `@register_crossencoder` - Cross-encoders

### Async-First

All agent `run()` and `stream()` methods are async generators.

### Factory Functions

Use factories from `tinygent.core.factory`:

- `build_agent()` - Build agent from config
- `build_llm()` - Build LLM from string
- `build_memory()` - Build memory from config
- `build_tool()` - Build tool from config
- `build_embedder()` - Build embedder from config
- `build_cross_encoder()` - Build cross-encoder from config
- `build_middleware()` - Build middleware from config
