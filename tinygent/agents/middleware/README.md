# Tinygent Middleware

Built-in middleware for agent lifecycle hooks.

## Available Middleware

### TinyToolCallLimiterMiddleware

Limits the number of tool calls per agent run. Can operate in two modes:
- **Global limiter**: Limits all tool calls when `tool_name=None`
- **Single tool limiter**: Limits specific tool by name when `tool_name` is set

**Features:**
- Limit all tools globally or specific tools only
- Hard block or soft limit behavior
- Per-run tracking with automatic cleanup
- Statistics tracking

**Parameters:**
- `tool_name` (str | None): Specific tool to limit, or None to limit all tools globally (default: None)
- `max_tool_calls` (int): Maximum number of tool calls allowed per run (default: 10)
- `hard_block` (bool): If True, blocks tool execution and returns error result. If False, allows execution but adds system message asking LLM to stop (default: True)

## Usage

### Option 1: Limit All Tools Globally

```python
from tinygent.agents.middleware import TinyToolCallLimiterMiddleware
from tinygent.agents import TinyMultiStepAgent
from tinygent.core.factory import build_llm

limiter = TinyToolCallLimiterMiddleware(max_tool_calls=5)

agent = TinyMultiStepAgent(
    llm=build_llm('openai:gpt-4o-mini'),
    tools=[web_search, calculator, database_query],
    middleware=[limiter],
)
```

### Option 2: Limit Specific Tool Only

```python
from tinygent.agents.middleware import TinyToolCallLimiterMiddleware

# Only limit expensive API calls
api_limiter = TinyToolCallLimiterMiddleware(
    tool_name='web_search',
    max_tool_calls=3
)

agent = TinyMultiStepAgent(
    llm=build_llm('openai:gpt-4o-mini'),
    tools=[web_search, calculator, database_query],
    middleware=[api_limiter],
)
```

### Option 3: Multiple Limiters for Different Tools

```python
from tinygent.agents.middleware import TinyToolCallLimiterMiddleware

middleware = [
    TinyToolCallLimiterMiddleware(tool_name='web_search', max_tool_calls=3),
    TinyToolCallLimiterMiddleware(tool_name='database_query', max_tool_calls=10),
    TinyToolCallLimiterMiddleware(tool_name='expensive_api', max_tool_calls=1),
]

agent = TinyMultiStepAgent(
    llm=build_llm('openai:gpt-4o-mini'),
    tools=[web_search, database_query, expensive_api, calculator],
    middleware=middleware,
)
```

### Option 4: Using Factory (Default Settings)

```python
from tinygent.core.factory import build_middleware, build_agent

agent = build_agent(
    'multistep',
    llm='openai:gpt-4o-mini',
    tools=[...],
    middleware=[build_middleware('tool_limiter')],
)
```

### Option 5: Using AgentConfig

```python
from tinygent.agents import TinyMultiStepAgentConfig
from tinygent.agents.middleware import TinyToolCallLimiterMiddleware

limiter = TinyToolCallLimiterMiddleware(max_tool_calls=3)

config = TinyMultiStepAgentConfig(
    llm='openai:gpt-4o-mini',
    tools=[...],
    middleware=[limiter],
)

agent = config.build()
```

### Getting Statistics

```python
limiter = TinyToolCallLimiterMiddleware(max_tool_calls=5)

agent = TinyMultiStepAgent(llm=..., tools=[...], middleware=[limiter])
result = agent.run('Some task')

stats = limiter.get_stats()
print(stats)
# {
#     'tool_name': None,
#     'max_tool_calls': 5,
#     'active_runs': 0,
#     'current_counts': {},
#     'runs_at_limit': 0
# }
```

## Creating Custom Middleware

```python
from tinygent.agents.middleware import TinyBaseMiddleware, register_middleware

@register_middleware('my_middleware')
class MyCustomMiddleware(TinyBaseMiddleware):
    def before_tool_call(self, *, run_id: str, tool, args):
        print(f'Calling {tool.info.name}')

    def on_answer(self, *, run_id: str, answer: str):
        print(f'Answer: {answer}')
```

## Middleware Hooks

All available hooks:

- `before_llm_call(run_id, llm_input)` - Before LLM API call
- `after_llm_call(run_id, llm_input, result)` - After LLM API call
- `before_tool_call(run_id, tool, args)` - Before tool execution
- `after_tool_call(run_id, tool, args, result)` - After tool execution
- `on_plan(run_id, plan)` - When agent creates a plan
- `on_reasoning(run_id, reasoning)` - When agent generates reasoning
- `on_tool_reasoning(run_id, reasoning)` - When tool generates reasoning
- `on_answer(run_id, answer)` - When agent provides final answer
- `on_answer_chunk(run_id, chunk, idx)` - For streaming answers
- `on_error(run_id, e)` - When an error occurs

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
