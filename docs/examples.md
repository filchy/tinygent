# Examples

Explore practical examples of Tinygent in action.

---

## Quick Examples

### Simple Weather Agent

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

### Calculator Agent

```python
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[add, multiply],
)

result = agent.run('What is (5 + 3) * 2?')
print(result)  # "The result is 16"
```

---

## Repository Examples

Tinygent includes comprehensive examples in the repository. All examples can be run with `uv run`:

### Basics

#### 1. Tool Usage

**Location**: `examples/tool-usage/main.py`

Demonstrates all tool decorator types:

- `@tool` - Simple tools
- `@register_tool` - Global registration with caching
- `@reasoning_tool` - Tools requiring reasoning
- `@jit_tool` - Just-in-time code generation

**Run:**

```bash
uv run examples/tool-usage/main.py
```

**Highlights:**

```python
# Pydantic model tools
@register_tool(use_cache=True)
def add(data: AddInput) -> int:
    """Adds two numbers together."""
    return data.a + data.b

# Regular parameter tools
@register_tool(use_cache=True)
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together."""
    return a * b

# Async tools
@tool
async def greet(data: GreetInput) -> str:
    """Greets a person by name."""
    return f'Hello, {data.name}!'

# Generator tools
@jit_tool(jit_instruction='Count from 1 to n, yielding each number.')
def count(data: CountInput):
    """Counts from 1 to n, yielding each number."""
    for i in range(1, data.n + 1):
        yield i

# Reasoning tools
@register_reasoning_tool(reasoning_prompt='Explain why you are performing this search.')
def search(data: SearchInput) -> str:
    """Search for something."""
    return f'Results for {data.query}'
```

---

#### 2. LLM Usage

**Location**: `examples/llm-usage/main.py`

Direct LLM usage without agents.

**Run:**

```bash
uv run examples/llm-usage/main.py
```

---

#### 3. Function Calling

**Location**: `examples/function-calling/main.py`

Demonstrates LLM function calling capabilities.

**Run:**

```bash
uv run examples/function-calling/main.py
```

---

### Memory Examples

#### 1. Buffer Chat Memory

**Location**: `examples/memory/basic-chat-memory/main.py`

Full conversation history with filtering.

**Run:**

```bash
uv run examples/memory/basic-chat-memory/main.py
```

**Highlights:**

```python
from tinygent.memory import BufferChatMemory

memory = BufferChatMemory()

# Save messages
memory.save_context(TinyHumanMessage(content='Hello'))
memory.save_context(TinyChatMessage(content='Hi there!'))

# Filter messages
memory._chat_history.add_filter(
    'only_human',
    lambda m: isinstance(m, TinyHumanMessage)
)
```

---

#### 2. Summary Buffer Memory

**Location**: `examples/memory/buffer-summary-memory/main.py`

Automatically summarizes old messages to save tokens.

**Run:**

```bash
uv run examples/memory/buffer-summary-memory/main.py
```

---

#### 3. Window Buffer Memory

**Location**: `examples/memory/buffer-window-chat-memory/main.py`

Keeps only the last N messages.

**Run:**

```bash
uv run examples/memory/buffer-window-chat-memory/main.py
```

---

#### 4. Combined Memory

**Location**: `examples/memory/combined-memory/main.py`

Combine multiple memory strategies.

**Run:**

```bash
uv run examples/memory/combined-memory/main.py
```

---

### Agent Examples

#### 1. ReAct Agent

**Location**: `examples/agents/react/main.py`

Full ReAct agent with middleware tracking the thought-action-observation cycle.

**Run:**

```bash
export OPENAI_API_KEY="your-key"
uv run examples/agents/react/main.py
```

**Highlights:**

```python
from tinygent.agents.react_agent import TinyReActAgent

# Custom middleware tracking ReAct cycles
class ReActCycleMiddleware(TinyBaseMiddleware):
    def on_reasoning(self, *, run_id: str, reasoning: str) -> None:
        print(f"THOUGHT: {reasoning}")

    def before_tool_call(self, *, run_id: str, tool, args) -> None:
        print(f"ACTION: {tool.info.name}({args})")

    def after_tool_call(self, *, run_id: str, tool, args, result) -> None:
        print(f"OBSERVATION: {result}")

agent = TinyReActAgent(
    llm=build_llm('openai:gpt-4o'),
    tools=[get_weather, get_best_destination],
    middleware=[ReActCycleMiddleware()],
    memory=BufferChatMemory(),
)
```

**Quick version**: `examples/agents/react/quick.py`

---

#### 2. Multi-Step Agent

**Location**: `examples/agents/multi-step/main.py`

Planning agent that creates a plan before execution.

**Run:**

```bash
uv run examples/agents/multi-step/main.py
```

**Quick version**: `examples/agents/multi-step/quick.py`

---

#### 3. Squad Agent

**Location**: `examples/agents/squad/main.py`

Multi-agent collaboration with specialized sub-agents.

**Run:**

```bash
uv run examples/agents/squad/main.py
```

**Quick version**: `examples/agents/squad/quick.py`

---

#### 4. MAP Agent

**Location**: `examples/agents/map/main.py`

Modular Agentic Planner with dynamic replanning.

**Run:**

```bash
uv run examples/agents/map/main.py
```

**Quick version**: `examples/agents/map/quick.py`

---

#### 5. Middleware Examples

**Location**: `examples/agents/middleware/main.py`

Three custom middleware examples:

1. **AnswerLoggingMiddleware** - Logs final answers
2. **LLMCallTimingMiddleware** - Tracks LLM call performance
3. **ToolCallAuditMiddleware** - Audits all tool executions

**Run:**

```bash
uv run examples/agents/middleware/main.py
```

**Highlights:**

```python
# Timing middleware
class LLMCallTimingMiddleware(TinyBaseMiddleware):
    def before_llm_call(self, *, run_id: str, llm_input) -> None:
        self.call_start_times[run_id] = time.time()

    def after_llm_call(self, *, run_id: str, llm_input, result) -> None:
        duration = time.time() - self.call_start_times[run_id]
        print(f"LLM call took {duration:.2f}s")

# Audit middleware
class ToolCallAuditMiddleware(TinyBaseMiddleware):
    def after_tool_call(self, *, run_id: str, tool, args, result) -> None:
        audit_entry = {
            'timestamp': time.time(),
            'tool': tool.info.name,
            'args': args,
            'result': str(result)[:100],
        }
        self.audit_log.append(audit_entry)

agent = TinyMultiStepAgent(
    llm=build_llm('openai:gpt-4o'),
    tools=[greet, add_numbers],
    middleware=[
        LLMCallTimingMiddleware(),
        ToolCallAuditMiddleware(),
        AnswerLoggingMiddleware(),
    ],
)
```

---

### Advanced Examples

#### Embeddings

**Location**: `examples/embeddings/main.py`

Generate vector embeddings for semantic search.

**Run:**

```bash
uv sync --extra openai --extra voyageai
uv run examples/embeddings/main.py
```

---

#### Cross-Encoder

**Location**: `examples/cross-encoder/main.py`

Re-rank search results using cross-encoders.

**Run:**

```bash
uv run examples/cross-encoder/main.py
```

---

#### Knowledge Graph

**Location**: `examples/knowledge-graph/main.py`

Build knowledge graphs with Neo4j.

**Run:**

```bash
uv sync --extra tiny_graph
# Requires Neo4j running
uv run examples/knowledge-graph/main.py
```

---

#### Tracing

**Location**: `examples/tracing/main.py`

Advanced tracing and observability.

**Run:**

```bash
uv run examples/tracing/main.py
```

---

#### Chat App

**Location**: `examples/chat-app/main.py`

Full chat application with FastAPI.

**Run:**

```bash
uv sync --extra tiny_chat
uv run examples/chat-app/main.py
```

Then open: `http://localhost:8000`

---

## Package Examples

### Brave Search

**Location**: `packages/tiny_brave/`

Real-world search tool using Brave Search API.

```python
# Install
uv sync --extra tiny_brave

# Use
from tiny_brave import brave_search

@register_tool
def web_search(query: str) -> str:
    """Search the web."""
    return brave_search(query)

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[web_search]
)
```

---

### Tiny Chat

**Location**: `packages/tiny_chat/`

Web-based chat interface for agents.

```bash
uv sync --extra tiny_chat
uv run -m tiny_chat
```

---

### Tiny Graph

**Location**: `packages/tiny_graph/`

Knowledge graph integration with Neo4j.

```bash
uv sync --extra tiny_graph
```

---

## Running Examples

### Prerequisites

```bash
# Clone repository
git clone git@github.com:filchy/tinygent.git
cd tinygent

# Setup environment
uv venv --seed .venv
source .venv/bin/activate

# Install dependencies
uv sync --extra openai
```

### Set API Keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export BRAVE_API_KEY="..."
```

### Run Any Example

```bash
uv run examples/agents/react/main.py
uv run examples/tool-usage/main.py
uv run examples/memory/basic-chat-memory/main.py
```

---

## Example Patterns

### Pattern 1: Development Workflow

```bash
# 1. Start with simple example
uv run examples/tool-usage/main.py

# 2. Try agent example
uv run examples/agents/react/quick.py

# 3. Add memory
uv run examples/memory/basic-chat-memory/main.py

# 4. Add middleware
uv run examples/agents/middleware/main.py

# 5. Build your own!
```

### Pattern 2: Learning Path

1. **Basics** → Tool usage, LLM usage
2. **Agents** → ReAct agent, Multi-step agent
3. **Memory** → Buffer memory, Window memory
4. **Advanced** → Middleware, Squad agents, MAP agents
5. **Production** → Chat app, Knowledge graphs, Tracing

---

## Contributing Examples

Have a cool example? Contribute it!

1. Create `examples/your-example/main.py`
2. Add README explaining the example
3. Submit a pull request

---

## Next Steps

- **[Getting Started](getting-started.md)**: Install and setup
- **[Core Concepts](concepts/agents.md)**: Learn the fundamentals
- **[Guides](guides/building-agents.md)**: Build your own agents

---

## Example Code Structure

All examples follow this structure:

```
examples/
├── agents/
│   ├── react/
│   │   ├── main.py          # Full example with middleware
│   │   ├── quick.py         # Minimal example
│   │   └── prompts.yaml     # Custom prompts
│   ├── multi-step/
│   ├── squad/
│   └── map/
├── memory/
│   ├── basic-chat-memory/
│   ├── buffer-summary-memory/
│   └── ...
├── tool-usage/
├── llm-usage/
└── ...
```

Each example is self-contained and can be run independently.
