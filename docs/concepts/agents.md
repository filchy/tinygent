# Agents

Agents are the core of Tinygent. They combine **LLMs**, **tools**, and **reasoning** to accomplish tasks autonomously.

---

## What is an Agent?

An agent is an autonomous system that:

1. **Receives** a task or question from the user
2. **Reasons** about how to solve it
3. **Uses tools** to gather information or take actions
4. **Iterates** until it has enough information
5. **Provides** a final answer

Unlike simple LLM calls, agents can:

- Break down complex tasks into steps
- Call external tools and APIs
- Remember conversation history
- Retry and self-correct
- Coordinate with other agents

---

## Agent Types

Tinygent provides 4 built-in agent types:

### 1. ReAct Agent

**Best for**: General-purpose tasks, tool-calling, reasoning loops

The ReAct (Reasoning + Acting) agent follows a thought-action-observation cycle:

```
Thought: I need to find the weather in Prague
Action: get_weather(location="Prague")
Observation: The weather is sunny, 75°F
Thought: I have the information I need
Final Answer: The weather in Prague is sunny with a high of 75°F.
```

**Usage:**

```python
from tinygent.core.factory import build_agent

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[get_weather, search_web],
    max_iterations=5,  # Max reasoning cycles
)

result = agent.run('What is the weather in Prague?')
```

**When to use:**

- Single-task execution
- Tool-heavy workflows
- When you need transparent reasoning

**Middleware Hooks Activated:**

- `before_llm_call` / `after_llm_call` - For LLM calls
- `before_tool_call` / `after_tool_call` - For tool executions
- `on_tool_reasoning` - When reasoning tools generate reasoning
- `on_answer` / `on_answer_chunk` - For final answers
- `on_error` - On any error

Note: ReAct agent does not use `on_plan` or `on_reasoning` hooks.

---

### 2. MultiStep Agent

**Best for**: Complex tasks requiring planning, step-by-step execution

The MultiStep agent creates a plan first, then executes it step by step:

```
1. Create Plan:
   - Step 1: Get weather in Prague
   - Step 2: Find best restaurant
   - Step 3: Make recommendation

2. Execute Steps:
   - Execute Step 1 → Result A
   - Execute Step 2 → Result B
   - Execute Step 3 → Result C

3. Synthesize Final Answer
```

**Usage:**

```python
from tinygent.agents.multi_step_agent import TinyMultiStepAgent
from tinygent.core.factory import build_llm

agent = TinyMultiStepAgent(
    llm=build_llm('openai:gpt-4o'),
    tools=[get_weather, find_restaurants, book_table],
    max_iterations=10,
)

result = agent.run(
    'Plan a dinner in Prague tonight - check weather, '
    'find a good restaurant, and book a table.'
)
```

**When to use:**

- Multi-step workflows
- Tasks requiring explicit planning
- When you need to see the plan before execution

**Middleware Hooks Activated:**

- `before_llm_call` / `after_llm_call` - For LLM calls
- `before_tool_call` / `after_tool_call` - For tool executions
- `on_plan` - When creating initial or updated plan
- `on_reasoning` - For agent reasoning steps
- `on_tool_reasoning` - When reasoning tools generate reasoning
- `on_answer` / `on_answer_chunk` - For final answers
- `on_error` - On any error

---

### 3. Squad Agent

**Best for**: Multi-agent collaboration, specialized roles

Squad agents coordinate multiple sub-agents, each with specialized capabilities:

```
Coordinator Agent
├── Research Agent (tools: web_search, wikipedia)
├── Analysis Agent (tools: calculator, data_analyzer)
└── Writer Agent (tools: text_formatter, summarizer)
```

**Usage:**

```python
from tinygent.agents.squad_agent import TinySquadAgent
from tinygent.core.factory import build_agent

# Create specialized agents
researcher = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[web_search, wikipedia],
)

analyst = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[calculator, analyze_data],
)

# Create squad
squad = TinySquadAgent(
    llm=build_llm('openai:gpt-4o'),
    agents=[researcher, analyst],
    max_iterations=5,
)

result = squad.run(
    'Research the GDP of Czech Republic and compare it to Poland'
)
```

**When to use:**

- Tasks requiring different specializations
- Divide-and-conquer strategies
- When you want parallel execution

**Middleware Hooks Activated:**

- `before_llm_call` / `after_llm_call` - For LLM calls (delegated to sub-agents)
- `before_tool_call` / `after_tool_call` - For tool executions (delegated to sub-agents)
- `on_answer` / `on_answer_chunk` - For final aggregated answers
- `on_error` - On any error

Note: Squad agent delegates most hooks to its sub-agents. Hook activation depends on sub-agent types.

---

### 4. MAP Agent (Modular Agentic Planner)

**Best for**: Complex workflows with dynamic replanning

MAP agents create modular plans and can adapt them based on execution results:

```
1. Initial Plan:
   Module A → Module B → Module C

2. Execute Module A
   → Result changes plan

3. Updated Plan:
   Module A → Module D → Module B → Module C

4. Continue execution with new plan
```

**Usage:**

```python
from tinygent.agents.map_agent import TinyMAPAgent

agent = TinyMAPAgent(
    llm=build_llm('openai:gpt-4o'),
    tools=[...],
    max_iterations=15,
)

result = agent.run('Complex multi-step task with potential replanning')
```

**When to use:**

- Highly dynamic tasks
- When the plan may need adjustment
- Research and exploration tasks

**Middleware Hooks Activated:**

- `before_llm_call` / `after_llm_call` - For LLM calls
- `before_tool_call` / `after_tool_call` - For tool executions
- `on_plan` - When creating search/action plans
- `on_answer` / `on_answer_chunk` - For final answers
- `on_error` - On any error

Note: MAP agent uses `on_plan` for action summaries but not `on_reasoning` or `on_tool_reasoning`.

---

## Agent Configuration

All agents support common configuration options:

### Using Config Objects

```python
from tinygent.agents.react_agent import ReactAgentConfig
from tinygent.memory import BufferChatMemory

config = ReactAgentConfig(
    llm='openai:gpt-4o-mini',
    tools=[get_weather],
    max_iterations=10,
    memory=BufferChatMemory(),
    temperature=0.7,
    stop_sequences=['STOP', 'END'],
)

agent = config.build()
```

### Using Factory Functions

```python
from tinygent.core.factory import build_agent

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[get_weather],
    max_iterations=10,
)
```

---

## Memory

Agents can remember conversation history using memory:

```python
from tinygent.memory import BufferChatMemory

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[get_weather],
    memory=BufferChatMemory(),
)

# First conversation
agent.run('What is the weather in Prague?')

# Second conversation - agent remembers context
agent.run('How about tomorrow?')
```

See [Memory](memory.md) for more details.

---

## Middleware

Customize agent behavior with middleware hooks:

```python
from tinygent.agents.middleware import TinyBaseMiddleware

class LoggingMiddleware(TinyBaseMiddleware):
    def on_reasoning(self, *, run_id: str, reasoning: str) -> None:
        print(f"[THOUGHT] {reasoning}")

    def before_tool_call(self, *, run_id: str, tool, args) -> None:
        print(f"[ACTION] {tool.info.name}({args})")

    def after_tool_call(self, *, run_id: str, tool, args, result) -> None:
        print(f"[OBSERVATION] {result}")

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[get_weather],
    middleware=[LoggingMiddleware()],
)
```

See [Middleware](middleware.md) for more details.

---

## Streaming Responses

All agents support async streaming:

```python
import asyncio

async def main():
    agent = build_agent('react', llm='openai:gpt-4o-mini', tools=[...])

    # Stream tokens as they're generated
    async for chunk in agent.run_stream('What is the weather?'):
        print(chunk, end='', flush=True)

asyncio.run(main())
```

---

## Choosing the Right Agent

| Task Type | Recommended Agent | Why |
|-----------|------------------|-----|
| Single API call | ReAct | Simple, fast, transparent |
| Multi-step workflow | MultiStep | Explicit planning |
| Requires specialists | Squad | Divide and conquer |
| Dynamic/exploratory | MAP | Adaptive replanning |
| General chatbot | ReAct + Memory | Conversational |
| Research task | MAP or Squad | Flexible exploration |

---

## Advanced Patterns

### Custom Prompts

Override default prompts:

```python
from tinygent.agents.react_agent import ReActPromptTemplate

custom_prompt = ReActPromptTemplate(
    system="You are a helpful AI assistant specialized in weather.",
    user_prefix="Human: ",
    assistant_prefix="Assistant: ",
    thought_prefix="Thinking: ",
    action_prefix="Action: ",
    observation_prefix="Observation: ",
)

agent = TinyReActAgent(
    llm=build_llm('openai:gpt-4o-mini'),
    tools=[get_weather],
    prompt_template=custom_prompt,
)
```

### Error Handling

Agents automatically handle tool errors:

```python
@tool
def risky_operation(data: str) -> str:
    """An operation that might fail."""
    if not data:
        raise ValueError("Data cannot be empty")
    return f"Processed: {data}"

# Agent will catch errors and try alternative approaches
agent = build_agent('react', llm='openai:gpt-4o-mini', tools=[risky_operation])
result = agent.run('Process this data')  # Handles errors gracefully
```

---

## Next Steps

- **[Tools](tools.md)**: Learn about tool types and decorators
- **[Memory](memory.md)**: Add conversation memory
- **[Middleware](middleware.md)**: Customize agent behavior
- **[Building Agents Guide](../guides/building-agents.md)**: Step-by-step guide

---

## Examples

Check out the examples directory:

- `examples/agents/react/main.py` - ReAct agent with tools
- `examples/agents/multi-step/main.py` - Multi-step planning
- `examples/agents/squad/main.py` - Multi-agent collaboration
- `examples/agents/map/main.py` - MAP agent with replanning
- `examples/agents/middleware/main.py` - Custom middleware
