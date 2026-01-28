# Middleware Example in TinyGent

This example demonstrates how to use **middleware** in TinyGent agents. Middleware allows you to monitor, customize, or intervene in the agent's lifecycle: LLM calls, tool calls, reasoning steps, final answers, and error handling.

## Quick Start

```bash
uv sync --extra openai

uv run examples/agents/middleware/main.py
```

## Concept

Middleware in TinyGent follows a class-based pattern. You create custom middleware by extending the `TinyBaseMiddleware` base class and overriding only the methods you need. Multiple middleware can be composed together using `TinyMiddlewareAgent`.

### Creating Custom Middleware

```python
from tinygent.agents.middleware import TinyBaseMiddleware

class LoggingMiddleware(TinyBaseMiddleware):
    def before_llm_call(self, *, run_id: str, llm_input: TinyLLMInput):
        print(f"[{run_id}] LLM call starting...")

    def after_tool_call(self, *, run_id: str, tool: AbstractTool, args: dict, result: Any):
        print(f"[{run_id}] Tool {tool.name} returned: {result}")

    def on_error(self, *, run_id: str, e: Exception):
        print(f"[{run_id}] Error occurred: {e}")
```

### Available Middleware Methods

| Method | Description |
|--------|-------------|
| `before_llm_call` | Triggered before an LLM is called |
| `after_llm_call` | Triggered after an LLM returns a result |
| `before_tool_call` | Triggered before a tool function is executed |
| `after_tool_call` | Triggered after a tool returns a result |
| `on_plan` | Triggered when the agent produces a plan |
| `on_reasoning` | Triggered when the agent produces reasoning steps |
| `on_tool_reasoning` | Triggered when the agent produces reasoning specific to tool usage |
| `on_answer` | Triggered when the agent produces its final answer |
| `on_answer_chunk` | Triggered for each chunk during streaming responses |
| `on_error` | Triggered when any error occurs |

---

## Composing Multiple Middleware

Use `TinyMiddlewareAgent` to combine multiple middleware instances:

```python
from tinygent.agents.middleware import TinyMiddlewareAgent
from tinygent.agents.middleware import TinyBaseMiddleware

class LoggingMiddleware(TinyBaseMiddleware):
    def on_answer(self, *, run_id: str, answer: str):
        print(f"Answer: {answer}")

class MetricsMiddleware(TinyBaseMiddleware):
    def before_llm_call(self, *, run_id: str, llm_input: TinyLLMInput):
        self.start_time = time.time()

    def after_llm_call(self, *, run_id: str, llm_input: TinyLLMInput, result: Any):
        elapsed = time.time() - self.start_time
        print(f"LLM call took {elapsed:.2f}s")

# Compose middleware
middleware_agent = TinyMiddlewareAgent([
    LoggingMiddleware(),
    MetricsMiddleware(),
])
```

---

## Why Middleware Matters

* **Debugging**: Inspect inputs, outputs, and tool usage.
* **Logging**: Stream detailed logs to your preferred system.
* **Metrics**: Track performance and usage statistics.
* **Customization**: Enforce additional checks or constraints.
* **Experimentation**: Compare how the agent behaves with different configurations.

---

## Example Flow

1. **Agent receives user input** → triggers `before_llm_call`.
2. **LLM generates reasoning or a plan** → triggers `on_plan` and/or `on_reasoning`.
3. **Agent decides to call a tool** → triggers `before_tool_call` → executes tool → triggers `after_tool_call` → triggers `on_tool_reasoning` if applicable.
4. **Agent gathers results and formulates answer** → triggers `after_llm_call`.
5. **Final answer is produced** → triggers `on_answer` (or `on_answer_chunk` for streaming).
6. **If an error occurs** at any stage → triggers `on_error`.

---

## Streaming + Middleware

Middleware works seamlessly with the streaming API. The `on_answer_chunk` method receives each chunk as it's generated:

```python
class StreamingMiddleware(TinyBaseMiddleware):
    def on_answer_chunk(self, *, run_id: str, chunk: str, idx: str):
        print(f"[chunk {idx}] {chunk}", end="")

async for chunk in agent.run_stream("Analyze the data and summarize it."):
    print("[STREAM]", chunk)
```

---

## Takeaway

Middleware provides a **transparent, composable window** into what your agent is doing at each step. The class-based pattern allows you to:

* Override only the methods you need
* Maintain state within your middleware instance
* Compose multiple middleware for different concerns
* Cleanly separate logging, metrics, and business logic
