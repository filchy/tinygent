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
from tinygent.agents.middleware.base import AgentMiddleware

class LoggingMiddleware(AgentMiddleware):
    def on_reasoning(self, *, run_id: str, reasoning: str) -> None:
        print(f"Thought: {reasoning}")

    def before_tool_call(self, *, run_id: str, tool, args) -> None:
        print(f"Calling: {tool.info.name}({args})")

    def after_tool_call(self, *, run_id: str, tool, args, result) -> None:
        print(f"Result: {result}")

    def on_answer(self, *, run_id: str, answer: str) -> None:
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
class AgentMiddleware:
    def on_start(self, *, run_id: str, task: str) -> None:
        """Called when agent starts processing a task."""
        pass

    def on_end(self, *, run_id: str) -> None:
        """Called when agent finishes processing."""
        pass

    def on_error(self, *, run_id: str, e: Exception) -> None:
        """Called when an error occurs."""
        pass
```

### LLM Calls

```python
class AgentMiddleware:
    def before_llm_call(self, *, run_id: str, llm_input) -> None:
        """Called before making an LLM API call."""
        pass

    def after_llm_call(self, *, run_id: str, llm_input, result) -> None:
        """Called after LLM API call completes."""
        pass
```

### Tool Calls

```python
class AgentMiddleware:
    def before_tool_call(self, *, run_id: str, tool, args: dict) -> None:
        """Called before executing a tool."""
        pass

    def after_tool_call(self, *, run_id: str, tool, args: dict, result) -> None:
        """Called after tool execution completes."""
        pass
```

### Reasoning and Answers

```python
class AgentMiddleware:
    def on_reasoning(self, *, run_id: str, reasoning: str) -> None:
        """Called when agent produces a thought/reasoning step."""
        pass

    def on_answer(self, *, run_id: str, answer: str) -> None:
        """Called when agent produces final answer."""
        pass

    def on_answer_chunk(self, *, run_id: str, chunk: str, idx: str) -> None:
        """Called for each streaming chunk of the answer."""
        pass
```

---

## Complete Example: ReAct Cycle Tracker

Track the Thought-Action-Observation cycle:

```python
from typing import Any
from tinygent.agents.middleware.base import AgentMiddleware

class ReActCycleMiddleware(AgentMiddleware):
    def __init__(self) -> None:
        self.cycles: list[dict[str, Any]] = []
        self.current_cycle: dict[str, Any] = {}
        self.iteration = 0

    def on_reasoning(self, *, run_id: str, reasoning: str) -> None:
        self.iteration += 1
        self.current_cycle = {
            'iteration': self.iteration,
            'thought': reasoning,
        }
        print(f"[Iteration {self.iteration}] {reasoning}")

    def before_tool_call(self, *, run_id: str, tool, args) -> None:
        self.current_cycle['action'] = {
            'tool': tool.info.name,
            'args': args,
        }
        print(f"[Iteration {self.iteration}] {tool.info.name}({args})")

    def after_tool_call(self, *, run_id: str, tool, args, result) -> None:
        self.current_cycle['observation'] = str(result)
        self.cycles.append(self.current_cycle.copy())
        print(f"[Iteration {self.iteration}] {result}")

    def on_answer(self, *, run_id: str, answer: str) -> None:
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

class TimingMiddleware(AgentMiddleware):
    def __init__(self) -> None:
        self.call_start_times: dict[str, float] = {}
        self.call_durations: list[float] = []

    def before_llm_call(self, *, run_id: str, llm_input) -> None:
        self.call_start_times[run_id] = time.time()

    def after_llm_call(self, *, run_id: str, llm_input, result) -> None:
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

class ToolAuditMiddleware(AgentMiddleware):
    def __init__(self, log_file: str = 'tool_audit.jsonl'):
        self.log_file = log_file

    def after_tool_call(self, *, run_id: str, tool, args, result) -> None:
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
class CostTrackingMiddleware(AgentMiddleware):
    def __init__(self):
        self.total_cost = 0.0
        self.costs_by_model = {}

        # Pricing per 1M tokens (example rates)
        self.pricing = {
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
            'gpt-4o': {'input': 2.50, 'output': 10.00},
        }

    def after_llm_call(self, *, run_id: str, llm_input, result) -> None:
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
class ErrorHandlingMiddleware(AgentMiddleware):
    def __init__(self):
        self.errors: list[dict] = []

    def on_error(self, *, run_id: str, e: Exception) -> None:
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
class StreamingDisplayMiddleware(AgentMiddleware):
    def on_answer_chunk(self, *, run_id: str, chunk: str, idx: str) -> None:
        # Print chunks as they arrive
        print(chunk, end='', flush=True)

    def on_answer(self, *, run_id: str, answer: str) -> None:
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
from tinygent.agents.middleware.base import register_middleware

@register_middleware('logging')
class LoggingMiddleware(AgentMiddleware):
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
class ConversationMetricsMiddleware(AgentMiddleware):
    def __init__(self):
        self.metrics = {
            'total_turns': 0,
            'total_tool_calls': 0,
            'total_tokens': 0,
            'avg_response_time': 0,
        }
        self.start_times = {}

    def on_start(self, *, run_id: str, task: str) -> None:
        self.start_times[run_id] = time.time()
        self.metrics['total_turns'] += 1

    def after_tool_call(self, *, run_id: str, tool, args, result) -> None:
        self.metrics['total_tool_calls'] += 1

    def after_llm_call(self, *, run_id: str, llm_input, result) -> None:
        self.metrics['total_tokens'] += result.usage.total_tokens

    def on_end(self, *, run_id: str) -> None:
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
class GodMiddleware(AgentMiddleware):
    def on_reasoning(self, ...):
        self.log()
        self.track_cost()
        self.send_analytics()
        self.update_ui()

# Good - Single responsibility
class LoggingMiddleware(AgentMiddleware):
    def on_reasoning(self, ...):
        self.log()

class CostMiddleware(AgentMiddleware):
    def on_reasoning(self, ...):
        self.track_cost()
```

### 2. Handle Errors Gracefully

```python
class SafeMiddleware(AgentMiddleware):
    def after_tool_call(self, *, run_id: str, tool, args, result) -> None:
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
class SlowMiddleware(AgentMiddleware):
    def before_llm_call(self, ...):
        time.sleep(5)  # Blocks!

# Good - Async for I/O
class AsyncMiddleware(AgentMiddleware):
    async def before_llm_call(self, ...):
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

### ToolCallLimiterMiddleware

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
from tinygent.agents.middleware import ToolCallLimiterMiddleware
from tinygent.agents import TinyMultiStepAgent
from tinygent.core.factory import build_llm

# Limit all tools to 5 calls
limiter = ToolCallLimiterMiddleware(max_tool_calls=5)

agent = TinyMultiStepAgent(
    llm=build_llm('openai:gpt-4o-mini'),
    tools=[search, calculator, database],
    middleware=[limiter],
)
```

**Limit Specific Tool:**

```python
# Only limit expensive API calls
api_limiter = ToolCallLimiterMiddleware(
    tool_name='web_search',
    max_tool_calls=3
)
```

**Hard Block vs Soft Limit:**

```python
# Hard block: returns error result when limit reached (default)
hard_limiter = ToolCallLimiterMiddleware(
    max_tool_calls=5,
    hard_block=True
)

# Soft limit: adds system message asking LLM to stop but allows execution
soft_limiter = ToolCallLimiterMiddleware(
    max_tool_calls=5,
    hard_block=False
)
```

**Multiple Limiters:**

```python
middleware = [
    ToolCallLimiterMiddleware(tool_name='web_search', max_tool_calls=3),
    ToolCallLimiterMiddleware(tool_name='database_query', max_tool_calls=10),
]
```

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
