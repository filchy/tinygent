# API Reference

Complete API reference for Tinygent.

---

## Core Factory Functions

### `build_agent()`

Build an agent from a string identifier.

```python
from tinygent.core.factory import build_agent

agent = build_agent(
    agent_type: str,
    llm: str | BaseLLM,
    tools: list[AbstractTool] = [],
    memory: BaseMemory | None = None,
    middleware: list[TinyBaseMiddleware] = [],
    max_iterations: int = 5,
    **kwargs
) -> BaseAgent
```

**Parameters:**

- `agent_type` (str): Agent type identifier (`'react'`, `'multi_step'`, `'squad'`, `'map'`)
- `llm` (str | BaseLLM): LLM identifier (e.g., `'openai:gpt-4o-mini'`) or LLM instance
- `tools` (list): List of tool functions
- `memory` (BaseMemory | None): Memory instance
- `middleware` (list): List of middleware instances
- `max_iterations` (int): Maximum reasoning iterations

**Returns:** Agent instance

**Example:**

```python
agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[get_weather],
    max_iterations=10
)
```

---

### `build_llm()`

Build an LLM from a provider string.

```python
from tinygent.core.factory import build_llm

llm = build_llm(
    llm_string: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs
) -> BaseLLM
```

**Parameters:**

- `llm_string` (str): Provider and model (`'provider:model'`)
- `temperature` (float): Sampling temperature (0.0-2.0)
- `max_tokens` (int | None): Maximum tokens to generate

**Returns:** LLM instance

**Example:**

```python
llm = build_llm('openai:gpt-4o-mini', temperature=0.3, max_tokens=1000)
```

---

### `build_memory()`

Build a memory instance from configuration.

```python
from tinygent.core.factory import build_memory

memory = build_memory(
    memory_type: str,
    **kwargs
) -> BaseMemory
```

**Parameters:**

- `memory_type` (str): Memory type identifier
- `**kwargs`: Memory-specific parameters

**Returns:** Memory instance

---

### `build_tool()`

Build a tool from configuration.

```python
from tinygent.core.factory import build_tool

tool = build_tool(
    tool_name: str,
    **kwargs
) -> AbstractTool
```

---

### `build_embedder()`

Build an embedder for vector embeddings.

```python
from tinygent.core.factory import build_embedder

embedder = build_embedder(
    embedder_string: str,
    **kwargs
) -> BaseEmbedder
```

**Example:**

```python
embedder = build_embedder('openai:text-embedding-3-small')
vectors = embedder.embed_documents(['Hello', 'World'])
```

---

## Agents

### ReActAgent

```python
from tinygent.agents.react_agent import TinyReActAgent

agent = TinyReActAgent(
    llm: BaseLLM,
    tools: list[AbstractTool],
    memory: BaseMemory | None = None,
    middleware: list[TinyBaseMiddleware] = [],
    max_iterations: int = 5,
    prompt_template: ReActPromptTemplate | None = None,
)
```

**Methods:**

- `run(task: str) -> str`: Execute task synchronously
- `run_stream(task: str) -> AsyncIterator[str]`: Execute with streaming
- `reset()`: Clear agent state

---

### MultiStepAgent

```python
from tinygent.agents.multi_step_agent import TinyMultiStepAgent

agent = TinyMultiStepAgent(
    llm: BaseLLM,
    tools: list[AbstractTool],
    memory: BaseMemory | None = None,
    middleware: list[TinyBaseMiddleware] = [],
    max_iterations: int = 10,
    prompt_template: MultiStepPromptTemplate | None = None,
)
```

---

### SquadAgent

```python
from tinygent.agents.squad_agent import TinySquadAgent

squad = TinySquadAgent(
    llm: BaseLLM,
    agents: list[BaseAgent],
    memory: BaseMemory | None = None,
    middleware: list[TinyBaseMiddleware] = [],
    max_iterations: int = 5,
)
```

---

### MAPAgent

```python
from tinygent.agents.map_agent import TinyMAPAgent

agent = TinyMAPAgent(
    llm: BaseLLM,
    tools: list[AbstractTool],
    memory: BaseMemory | None = None,
    middleware: list[TinyBaseMiddleware] = [],
    max_iterations: int = 15,
)
```

---

## Tool Decorators

### `@tool`

Create a simple tool.

```python
from tinygent.tools import tool

@tool
def my_function(param: str) -> str:
    """Function description."""
    return result
```

---

### `@register_tool`

Create and globally register a tool.

```python
from tinygent.tools import register_tool

@register_tool(use_cache: bool = False, hidden: bool = False)
def my_function(param: str) -> str:
    """Function description."""
    return result
```

**Parameters:**

- `use_cache` (bool): Enable result caching
- `hidden` (bool): Hide from default tool listings

---

### `@reasoning_tool`

Create a tool requiring reasoning.

```python
from tinygent.tools import register_reasoning_tool

@register_reasoning_tool(reasoning_prompt: str)
def my_function(param: str) -> str:
    """Function description."""
    return result
```

**Parameters:**

- `reasoning_prompt` (str): Prompt for agent reasoning

---

### `@jit_tool`

Create a just-in-time code generation tool.

```python
from tinygent.tools import jit_tool

@jit_tool(jit_instruction: str)
def my_function(param: str):
    """Function description."""
    yield result
```

**Parameters:**

- `jit_instruction` (str): Code generation instructions

---

## Memory

### BufferChatMemory

```python
from tinygent.memory import BufferChatMemory

memory = BufferChatMemory()
```

**Methods:**

- `save_context(message: TinyMessage)`: Save a message
- `load_variables() -> list[TinyMessage]`: Load all messages
- `clear()`: Clear all messages

---

### SummaryBufferMemory

```python
from tinygent.memory import SummaryBufferMemory

memory = SummaryBufferMemory(
    llm: BaseLLM,
    max_token_limit: int = 1000,
)
```

---

### WindowBufferMemory

```python
from tinygent.memory import WindowBufferMemory

memory = WindowBufferMemory(
    window_size: int = 4,
)
```

---

### CombinedMemory

```python
from tinygent.memory import CombinedMemory

memory = CombinedMemory(
    memories: dict[str, BaseMemory],
)
```

---

## Middleware

### Base Middleware

```python
from tinygent.agents.middleware import TinyBaseMiddleware

class CustomMiddleware(TinyBaseMiddleware):
    def on_start(self, *, run_id: str, task: str) -> None:
        pass

    def on_end(self, *, run_id: str) -> None:
        pass

    def on_error(self, *, run_id: str, e: Exception) -> None:
        pass

    def before_llm_call(self, *, run_id: str, llm_input) -> None:
        pass

    def after_llm_call(self, *, run_id: str, llm_input, result) -> None:
        pass

    def before_tool_call(self, *, run_id: str, tool, args: dict) -> None:
        pass

    def after_tool_call(self, *, run_id: str, tool, args: dict, result) -> None:
        pass

    def on_reasoning(self, *, run_id: str, reasoning: str) -> None:
        pass

    def on_answer(self, *, run_id: str, answer: str) -> None:
        pass

    def on_answer_chunk(self, *, run_id: str, chunk: str, idx: str) -> None:
        pass
```

---

### Register Middleware

```python
from tinygent.agents.middleware import TinyBaseMiddleware

@register_middleware('my_middleware')
class MyMiddleware(TinyBaseMiddleware):
    # ...
```

---

## Messages

### Message Types

```python
from tinygent.core.datamodels.messages import (
    TinyHumanMessage,
    TinyChatMessage,
    TinySystemMessage,
    TinyPlanMessage,
    TinyToolMessage,
)

# Create messages
human_msg = TinyHumanMessage(content="Hello")
ai_msg = TinyChatMessage(content="Hi there!")
system_msg = TinySystemMessage(content="You are a helpful assistant")
```

---

## Data Models

### TinyModel

Base class for Pydantic models.

```python
from pydantic import Field
from tinygent.core.types import TinyModel

class MyInput(TinyModel):
    name: str = Field(..., description='User name')
    age: int = Field(..., ge=0, description='User age')
```

---

## Runtime Registry

### Tool Catalog

```python
from tinygent.core.runtime.tool_catalog import GlobalToolCatalog

# Get catalog
catalog = GlobalToolCatalog().get_active_catalog()

# List tools
tools = catalog.list_tools()

# Get specific tool
tool = catalog.get_tool('tool_name')

# Call tool
result = tool(param='value')
```

---

### Global Registry

```python
from tinygent.core.runtime.global_registry import (
    get_registered_agents,
    get_registered_llms,
    get_registered_tools,
    get_registered_memories,
)

# Get all registered components
agents = get_registered_agents()
llms = get_registered_llms()
tools = get_registered_tools()
memories = get_registered_memories()
```

---

## LLM Usage

### Direct LLM Calls

```python
from tinygent.core.factory import build_llm

llm = build_llm('openai:gpt-4o-mini')

# Synchronous
response = llm.generate(prompt="What is AI?")
print(response.content)

# Asynchronous
response = await llm.agenerate(prompt="What is AI?")

# Streaming
async for chunk in llm.stream(prompt="Tell me a story"):
    print(chunk, end='', flush=True)
```

---

## Logging

### Setup Logger

```python
from tinygent.logging import setup_logger

# Debug level
logger = setup_logger('debug')

# Info level
logger = setup_logger('info')

# Use logger
logger.info("Message")
logger.debug("Debug message")
logger.error("Error message")
```

---

## Utilities

### Color Printer

```python
from tinygent.utils import TinyColorPrinter

# Predefined colors
print(TinyColorPrinter.success("Success!"))
print(TinyColorPrinter.error("Error!"))
print(TinyColorPrinter.warning("Warning!"))
print(TinyColorPrinter.info("Info"))

# Custom color
print(TinyColorPrinter.custom("Label", "Message", color="CYAN"))
```

---

### YAML Loader

```python
from tinygent.utils import tiny_yaml_load

config = tiny_yaml_load('config.yaml')
```

---

## Type Hints

### Common Types

```python
from typing import List, Dict, Optional, Any
from tinygent.core.types import TinyModel
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.datamodels.messages import TinyMessage
from tinygent.agents.base import BaseAgent
from tinygent.memory import BaseMemory
from tinygent.llms.base import BaseLLM
```

---

## Configuration Classes

### ReactAgentConfig

```python
from tinygent.agents.react_agent import ReactAgentConfig

config = ReactAgentConfig(
    llm='openai:gpt-4o-mini',
    tools=[...],
    max_iterations=5,
    memory=memory,
    middleware=[...],
)

agent = config.build()
```

---

## Next Steps

- **[Getting Started](getting-started.md)**: Setup guide
- **[Core Concepts](concepts/agents.md)**: Learn fundamentals
- **[Examples](examples.md)**: See code examples

---

## Package Structure

```
tinygent/
├── agents/          # Agent implementations
├── core/            # Core functionality
│   ├── datamodels/  # Data models
│   ├── factory/     # Factory functions
│   ├── prompts/     # Prompt templates
│   ├── runtime/     # Runtime registries
│   └── types/       # Type definitions
├── llms/            # LLM integrations
├── memory/          # Memory implementations
├── tools/           # Tool decorators
├── cli/             # CLI commands
└── utils/           # Utilities

packages/
├── tiny_openai/     # OpenAI integration
├── tiny_anthropic/  # Anthropic integration
├── tiny_mistralai/  # Mistral integration
├── tiny_gemini/     # Gemini integration
├── tiny_brave/      # Brave search
├── tiny_chat/       # Chat UI
└── tiny_graph/      # Neo4j graphs
```

---

For detailed implementation, see the source code in the repository.
