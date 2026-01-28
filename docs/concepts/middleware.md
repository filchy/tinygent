# Middleware

Middleware allows you to customize agent behavior by hooking into key events during agent execution.

---

## What is Middleware?

Middleware provides **lifecycle hooks** for agents:

- **Before/after LLM calls**: Log prompts, track costs
- **Before/after tool calls**: Audit, validate, cache
- **On reasoning**: Monitor agent thoughts
- **On errors**: Handle failures gracefully
- **On answers**: Process final outputs

Think of middleware as **event listeners** for agent operations.

---

## Basic Example

```python
from tinygent.agents.middleware import TinyBaseMiddleware

class LoggingMiddleware(TinyBaseMiddleware):
    def on_reasoning(self, *, run_id: str, reasoning: str, **kwargs) -> None:
        print(f"Thought: {reasoning}")

    def before_tool_call(self, *, run_id: str, tool, args, **kwargs) -> None:
        print(f"Calling: {tool.info.name}({args})")

    def after_tool_call(self, *, run_id: str, tool, args, result, **kwargs) -> None:
        print(f"Result: {result}")

    def on_answer(self, *, run_id: str, answer: str, **kwargs) -> None:
        print(f"Final Answer: {answer}")

# Use it
from tinygent.core.factory import build_agent

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[get_weather],
    middleware=[LoggingMiddleware()]
)

result = agent.run('What is the weather in Prague?')
```

**Output:**

```
Thought: I need to check the weather in Prague
Calling: get_weather({'location': 'Prague'})
Result: The weather in Prague is sunny with a high of 75°F
Thought: I have the information needed
Final Answer: The weather in Prague is sunny with a high of 75°F.
```

---

## Middleware Hooks

### Agent Lifecycle

```python
class TinyBaseMiddleware:
    def on_start(self, *, run_id: str, task: str, **kwargs) -> None:
        """Called when agent starts processing a task."""
        pass

    def on_end(self, *, run_id: str, **kwargs) -> None:
        """Called when agent finishes processing."""
        pass

    def on_error(self, *, run_id: str, e: Exception, **kwargs) -> None:
        """Called when an error occurs."""
        pass
```

### LLM Calls

```python
class TinyBaseMiddleware:
    def before_llm_call(self, *, run_id: str, llm_input, **kwargs) -> None:
        """Called before making an LLM API call."""
        pass

    def after_llm_call(self, *, run_id: str, llm_input, result, **kwargs) -> None:
        """Called after LLM API call completes."""
        pass
```

### Tool Calls

```python
class TinyBaseMiddleware:
    def before_tool_call(self, *, run_id: str, tool, args: dict, **kwargs) -> None:
        """Called before executing a tool."""
        pass

    def after_tool_call(self, *, run_id: str, tool, args: dict, result, **kwargs) -> None:
        """Called after tool execution completes."""
        pass
```

### Reasoning and Answers

```python
class TinyBaseMiddleware:
    def on_reasoning(self, *, run_id: str, reasoning: str, **kwargs) -> None:
        """Called when agent produces a thought/reasoning step."""
        pass

    def on_answer(self, *, run_id: str, answer: str, **kwargs) -> None:
        """Called when agent produces final answer."""
        pass

    def on_answer_chunk(self, *, run_id: str, chunk: str, idx: str, **kwargs) -> None:
        """Called for each streaming chunk of the answer."""
        pass
```

---

## Complete Example: ReAct Cycle Tracker

Track the Thought-Action-Observation cycle:

```python
from typing import Any
from tinygent.agents.middleware import TinyBaseMiddleware

class ReActCycleMiddleware(TinyBaseMiddleware):
    def __init__(self) -> None:
        self.cycles: list[dict[str, Any]] = []
        self.current_cycle: dict[str, Any] = {}
        self.iteration = 0

    def on_reasoning(self, *, run_id: str, reasoning: str, **kwargs) -> None:
        self.iteration += 1
        self.current_cycle = {
            'iteration': self.iteration,
            'thought': reasoning,
        }
        print(f"[Iteration {self.iteration}] {reasoning}")

    def before_tool_call(self, *, run_id: str, tool, args, **kwargs) -> None:
        self.current_cycle['action'] = {
            'tool': tool.info.name,
            'args': args,
        }
        print(f"[Iteration {self.iteration}] {tool.info.name}({args})")

    def after_tool_call(self, *, run_id: str, tool, args, result, **kwargs) -> None:
        self.current_cycle['observation'] = str(result)
        self.cycles.append(self.current_cycle.copy())
        print(f"[Iteration {self.iteration}] {result}")

    def on_answer(self, *, run_id: str, answer: str, **kwargs) -> None:
        print(f"Final Answer after {self.iteration} iterations")

    def get_summary(self) -> dict[str, Any]:
        """Get execution summary."""
        tools_used = [
            c.get('action', {}).get('tool')
            for c in self.cycles
            if 'action' in c
        ]
        return {
            'total_iterations': self.iteration,
            'completed_cycles': len(self.cycles),
            'tools_used': list(set(tools_used)),
        }

# Usage
middleware = ReActCycleMiddleware()

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[...],
    middleware=[middleware]
)

result = agent.run('Complex task')

# Get insights
print(middleware.get_summary())
# {
#   'total_iterations': 3,
#   'completed_cycles': 3,
#   'tools_used': ['get_weather', 'search_web']
# }
```

---

## Use Cases

### 1. Performance Monitoring

Track LLM call timing:

```python
import time

class TimingMiddleware(TinyBaseMiddleware):
    def __init__(self) -> None:
        self.call_start_times: dict[str, float] = {}
        self.call_durations: list[float] = []

    def before_llm_call(self, *, run_id: str, llm_input, **kwargs) -> None:
        self.call_start_times[run_id] = time.time()

    def after_llm_call(self, *, run_id: str, llm_input, result, **kwargs) -> None:
        start = self.call_start_times.pop(run_id, None)
        if start:
            duration = time.time() - start
            self.call_durations.append(duration)
            print(f"LLM call took {duration:.2f}s")

    def get_stats(self) -> dict:
        if not self.call_durations:
            return {'avg': 0, 'total': 0}

        return {
            'total_calls': len(self.call_durations),
            'avg_duration': sum(self.call_durations) / len(self.call_durations),
            'total_duration': sum(self.call_durations),
            'min': min(self.call_durations),
            'max': max(self.call_durations),
        }
```

### 2. Tool Auditing

Log all tool calls for compliance:

```python
import json
from datetime import datetime

class ToolAuditMiddleware(TinyBaseMiddleware):
    def __init__(self, log_file: str = 'tool_audit.jsonl'):
        self.log_file = log_file

    def after_tool_call(self, *, run_id: str, tool, args, result, **kwargs) -> None:
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'run_id': run_id,
            'tool': tool.info.name,
            'args': args,
            'result': str(result)[:200],  # Truncate
        }

        # Append to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')
```

### 3. Cost Tracking

Track API costs:

```python
class CostTrackingMiddleware(TinyBaseMiddleware):
    def __init__(self):
        self.total_cost = 0.0
        self.costs_by_model = {}

        # Pricing per 1M tokens (example rates)
        self.pricing = {
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
            'gpt-4o': {'input': 2.50, 'output': 10.00},
        }

    def after_llm_call(self, *, run_id: str, llm_input, result, **kwargs) -> None:
        model = result.model  # e.g., 'gpt-4o-mini'
        input_tokens = result.usage.prompt_tokens
        output_tokens = result.usage.completion_tokens

        if model in self.pricing:
            rates = self.pricing[model]
            cost = (
                (input_tokens / 1_000_000) * rates['input'] +
                (output_tokens / 1_000_000) * rates['output']
            )

            self.total_cost += cost
            self.costs_by_model[model] = self.costs_by_model.get(model, 0) + cost

            print(f"Cost for this call: ${cost:.6f}")

    def get_total_cost(self) -> float:
        return self.total_cost
```

### 4. Error Handling

Gracefully handle errors:

```python
class ErrorHandlingMiddleware(TinyBaseMiddleware):
    def __init__(self):
        self.errors: list[dict] = []

    def on_error(self, *, run_id: str, e: Exception, **kwargs) -> None:
        error_info = {
            'run_id': run_id,
            'error_type': type(e).__name__,
            'message': str(e),
            'timestamp': datetime.now().isoformat(),
        }

        self.errors.append(error_info)

        # Log to file
        with open('errors.log', 'a') as f:
            f.write(f"[{error_info['timestamp']}] {error_info['error_type']}: {error_info['message']}\n")

        # Send alert (Slack, email, etc.)
        # self.send_alert(error_info)
```

### 5. Streaming Display

Pretty-print streaming output:

```python
class StreamingDisplayMiddleware(TinyBaseMiddleware):
    def on_answer_chunk(self, *, run_id: str, chunk: str, idx: str, **kwargs) -> None:
        # Print chunks as they arrive
        print(chunk, end='', flush=True)

    def on_answer(self, *, run_id: str, answer: str, **kwargs) -> None:
        # Print newline after complete answer
        print("\n")
```

---

## Registering Middleware

### Local Registration

```python
# Use directly in agent
agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[...],
    middleware=[LoggingMiddleware(), TimingMiddleware()]
)
```

### Global Registration

Make middleware reusable:

```python
from tinygent.agents.middleware import TinyBaseMiddleware

@register_middleware('logging')
class LoggingMiddleware(TinyBaseMiddleware):
    # ... implementation ...

# Later, build from registry
from tinygent.core.factory import build_middleware

middleware = build_middleware('logging')
agent = build_agent('react', llm='...', middleware=[middleware])
```

---

## Multiple Middleware

Chain multiple middleware together:

```python
timing = TimingMiddleware()
logging = LoggingMiddleware()
cost_tracker = CostTrackingMiddleware()
error_handler = ErrorHandlingMiddleware()

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[...],
    middleware=[timing, logging, cost_tracker, error_handler]
)

result = agent.run('Complex task')

# Get insights from each
print(f"Stats: {timing.get_stats()}")
print(f"Cost: ${cost_tracker.get_total_cost():.4f}")
print(f"Errors: {len(error_handler.errors)}")
```

---

## Advanced: State Management

Middleware can maintain state across calls:

```python
class ConversationMetricsMiddleware(TinyBaseMiddleware):
    def __init__(self):
        self.metrics = {
            'total_turns': 0,
            'total_tool_calls': 0,
            'total_tokens': 0,
            'avg_response_time': 0,
        }
        self.start_times = {}

    def on_start(self, *, run_id: str, task: str, **kwargs) -> None:
        self.start_times[run_id] = time.time()
        self.metrics['total_turns'] += 1

    def after_tool_call(self, *, run_id: str, tool, args, result, **kwargs) -> None:
        self.metrics['total_tool_calls'] += 1

    def after_llm_call(self, *, run_id: str, llm_input, result, **kwargs) -> None:
        self.metrics['total_tokens'] += result.usage.total_tokens

    def on_end(self, *, run_id: str, **kwargs) -> None:
        start = self.start_times.pop(run_id, None)
        if start:
            duration = time.time() - start
            # Update rolling average
            prev_avg = self.metrics['avg_response_time']
            n = self.metrics['total_turns']
            self.metrics['avg_response_time'] = (prev_avg * (n-1) + duration) / n

    def get_metrics(self) -> dict:
        return self.metrics
```

---

## Best Practices

### 1. Keep Middleware Focused

```python
# Bad - Does too much
class GodMiddleware(TinyBaseMiddleware):
    def on_reasoning(self, *, run_id, reasoning, **kwargs):
        self.log()
        self.track_cost()
        self.send_analytics()
        self.update_ui()

# Good - Single responsibility
class LoggingMiddleware(TinyBaseMiddleware):
    def on_reasoning(self, *, run_id, reasoning, **kwargs):
        self.log()

class CostMiddleware(TinyBaseMiddleware):
    def on_reasoning(self, *, run_id, reasoning, **kwargs):
        self.track_cost()
```

### 2. Handle Errors Gracefully

```python
class SafeMiddleware(TinyBaseMiddleware):
    def after_tool_call(self, *, run_id: str, tool, args, result, **kwargs) -> None:
        try:
            # Your logic
            self.process(result)
        except Exception as e:
            # Don't crash the agent
            print(f"Middleware error: {e}")
```

### 3. Avoid Blocking Operations

```python
# Bad - Blocks agent execution
class SlowMiddleware(TinyBaseMiddleware):
    def before_llm_call(self, *, run_id, llm_input, **kwargs):
        time.sleep(5)  # Blocks!

# Good - Async for I/O
class AsyncMiddleware(TinyBaseMiddleware):
    async def before_llm_call(self, *, run_id, llm_input, **kwargs):
        await async_operation()
```

---

## Middleware vs. Tools

**Use middleware for:**

- Logging and monitoring
- Cost tracking
- Performance metrics
- Error handling
- Auditing

**Use tools for:**

- External API calls
- Data retrieval
- Computations
- Actions (sending emails, etc.)

---

## Agent-Specific Hook Activation

Different agent types activate different hooks based on their implementation:

### TinyMultiStepAgent

Activates:
- `before_llm_call` / `after_llm_call` - For LLM calls
- `before_tool_call` / `after_tool_call` - For tool executions
- `on_plan` - When creating initial or updated plan
- `on_reasoning` - For agent reasoning steps
- `on_tool_reasoning` - When reasoning tools generate reasoning
- `on_answer` / `on_answer_chunk` - For final answers
- `on_error` - On any error

### TinyReactAgent

Activates:
- `before_llm_call` / `after_llm_call` - For LLM calls
- `before_tool_call` / `after_tool_call` - For tool executions
- `on_tool_reasoning` - When reasoning tools generate reasoning
- `on_answer` / `on_answer_chunk` - For final answers
- `on_error` - On any error

Note: React agent does not use `on_plan` or `on_reasoning` hooks.

### TinyMAPAgent

Activates:
- `before_llm_call` / `after_llm_call` - For LLM calls
- `before_tool_call` / `after_tool_call` - For tool executions
- `on_plan` - When creating search/action plans
- `on_answer` / `on_answer_chunk` - For final answers
- `on_error` - On any error

Note: MAP agent uses `on_plan` for action summaries but not `on_reasoning` or `on_tool_reasoning`.

### TinySquadAgent

Activates:
- `before_llm_call` / `after_llm_call` - For LLM calls (delegated to sub-agents)
- `before_tool_call` / `after_tool_call` - For tool executions (delegated to sub-agents)
- `on_answer` / `on_answer_chunk` - For final aggregated answers
- `on_error` - On any error

Note: Squad agent delegates most hooks to its sub-agents. Hook activation depends on sub-agent types.

---

## Built-in Middleware

Tinygent provides ready-to-use middleware for common use cases.

### TinyToolCallLimiterMiddleware

Limits the number of tool calls per agent run. Can operate in two modes:
- **Global limiter**: Limits all tool calls when `tool_name=None`
- **Single tool limiter**: Limits specific tool by name when `tool_name` is set

When the limit is reached, the behavior depends on `hard_block`:
- **hard_block=True**: Blocks tool execution and returns error result
- **hard_block=False**: Allows execution but adds system message asking LLM to stop

**Features:**
- Limit all tools globally or specific tools individually
- Hard block or soft limit behavior
- Per-run tracking with automatic cleanup
- Statistics tracking

**Basic Usage:**

```python
from tinygent.agents.middleware import TinyToolCallLimiterMiddleware
from tinygent.agents import TinyMultiStepAgent
from tinygent.core.factory import build_llm

# Limit all tools to 5 calls
limiter = TinyToolCallLimiterMiddleware(max_tool_calls=5)

agent = TinyMultiStepAgent(
    llm=build_llm('openai:gpt-4o-mini'),
    tools=[search, calculator, database],
    middleware=[limiter],
)
```

**Limit Specific Tool:**

```python
# Only limit expensive API calls
api_limiter = TinyToolCallLimiterMiddleware(
    tool_name='web_search',
    max_tool_calls=3
)
```

**Hard Block vs Soft Limit:**

```python
# Hard block: returns error result when limit reached (default)
hard_limiter = TinyToolCallLimiterMiddleware(
    max_tool_calls=5,
    hard_block=True
)

# Soft limit: adds system message asking LLM to stop but allows execution
soft_limiter = TinyToolCallLimiterMiddleware(
    max_tool_calls=5,
    hard_block=False
)
```

**Multiple Limiters:**

```python
middleware = [
    TinyToolCallLimiterMiddleware(tool_name='web_search', max_tool_calls=3),
    TinyToolCallLimiterMiddleware(tool_name='database_query', max_tool_calls=10),
]
```

**Using Config Factory:**

```python
from tinygent.agents.middleware import TinyToolCallLimiterMiddlewareConfig

# Create via config
config = TinyToolCallLimiterMiddlewareConfig(
    tool_name='web_search',
    max_tool_calls=5,
    hard_block=True
)

limiter = config.build()
```

**Factory Configuration Options:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `Literal['tool_limiter']` | `'tool_limiter'` | Type identifier (frozen) |
| `tool_name` | `str \| None` | `None` | Specific tool to limit. `None` = limit all tools globally |
| `max_tool_calls` | `int` | `10` | Maximum number of tool calls allowed per run |
| `hard_block` | `bool` | `True` | Whether to hard block (`True`) or soft limit (`False`) tool calls |

**Getting Statistics:**

```python
stats = limiter.get_stats()
# {
#     'tool_name': 'web_search',
#     'max_tool_calls': 3,
#     'hard_block': True,
#     'active_runs': 0,
#     'current_counts': {},
#     'runs_at_limit': 0
# }
```

---

### TinyLLMToolSelectorMiddleware

Intelligently selects the most relevant subset of tools for each LLM call using a smaller LLM. This middleware is especially useful when you have many tools available but want to reduce context size and improve performance by only providing the most relevant tools to the main agent.

**Features:**
- Uses a dedicated LLM to select relevant tools based on conversation context
- Reduces token usage by limiting tools sent to the main LLM
- Supports always-include list for critical tools
- Configurable maximum tools limit
- Automatic prompt template management

**How It Works:**
1. Before each LLM call, the middleware analyzes the conversation context
2. Uses a selection LLM to determine which tools are most relevant
3. Filters the tool list to only include selected tools
4. The main agent LLM receives only the relevant subset

**Basic Usage:**

```python
from tinygent.agents.middleware import TinyLLMToolSelectorMiddleware
from tinygent.agents import TinyMultiStepAgent
from tinygent.core.factory import build_llm

# Use fast model for tool selection
selector = TinyLLMToolSelectorMiddleware(
    llm=build_llm('openai:gpt-4o-mini'),
    max_tools=5
)

agent = TinyMultiStepAgent(
    llm=build_llm('openai:gpt-4o'),
    tools=[search, calculator, weather, database, email, calendar, notes],
    middleware=[selector],
)
```

**Always Include Critical Tools:**

```python
# Ensure certain tools are always available
selector = TinyLLMToolSelectorMiddleware(
    llm=build_llm('openai:gpt-4o-mini'),
    max_tools=5,
    always_include=['search', 'calculator']
)
```

**Custom Prompt Template:**

```python
from tinygent.prompts.middleware import LLMToolSelectorPromptTemplate

custom_prompt = LLMToolSelectorPromptTemplate(
    system="You are a tool selection expert. Select only the most relevant tools.",
    user="Available tools:\n{{ tools }}\n\nSelect the best tools for the current task."
)

selector = TinyLLMToolSelectorMiddleware(
    llm=build_llm('openai:gpt-4o-mini'),
    prompt_template=custom_prompt,
    max_tools=3
)
```

**Using Config Factory:**

```python
from tinygent.agents.middleware import TinyLLMToolSelectorMiddlewareConfig

# Create via config
config = TinyLLMToolSelectorMiddlewareConfig(
    llm='openai:gpt-4o-mini',
    max_tools=5,
    always_include=['search', 'calculator']
)

selector = config.build()
```

**Factory Configuration Options:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `Literal['llm_tool_selector']` | `'llm_tool_selector'` | Type identifier (frozen) |
| `llm` | `AbstractLLMConfig \| AbstractLLM` | Required | LLM to use for tool selection. Can be a string like `'openai:gpt-4o-mini'` or an LLM instance |
| `prompt_template` | `LLMToolSelectorPromptTemplate` | Default prompt | Template for tool selection prompt. Contains `system` and `user` fields |
| `max_tools` | `int \| None` | `None` | Maximum number of tools to select. `None` = no limit |
| `always_include` | `list[str] \| None` | `None` | List of tool names to always include in selection |

**Advanced Example:**

```python
# Combine with tool limiter
selector = TinyLLMToolSelectorMiddleware(
    llm=build_llm('openai:gpt-4o-mini'),
    max_tools=5,
    always_include=['search']
)

limiter = TinyToolCallLimiterMiddleware(
    max_tool_calls=10,
    hard_block=False
)

agent = TinyMultiStepAgent(
    llm=build_llm('openai:gpt-4o'),
    tools=[search, calculator, weather, database, email, calendar, notes, api_call],
    middleware=[selector, limiter],
)
```

**Benefits:**
- **Reduced Token Usage**: Only send relevant tools to the main LLM
- **Improved Performance**: Faster LLM responses with smaller context
- **Better Focus**: Agent focuses on appropriate tools for the task
- **Cost Savings**: Fewer tokens = lower API costs

**When to Use:**
- You have 10+ tools available
- Tools have large descriptions
- You want to optimize token usage
- You need dynamic tool selection based on context

---

## Next Steps

- **[Agents](agents.md)**: Use middleware with agents
- **[Examples](../examples.md)**: See middleware examples
- **[Building Agents Guide](../guides/building-agents.md)**: Build custom agents with middleware

---

## Examples

Check out:

- `examples/agents/middleware/main.py` - Multiple middleware examples
- `examples/agents/middleware/tool_limiter_example.py` - Tool call limiting examples
- `examples/agents/react/main.py` - ReAct cycle tracking
- `examples/tracing/main.py` - Advanced tracing middleware
