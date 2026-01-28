# TinyReActAgent Example - Reasong + Act Agent

This example demonstrates how to build and run a **ReAct style agent** (`TinyReActAgent`) using `tinygent`.
The agent alternates between **reasoning** and **acting**, while keeping track of tools and conversation history.

```mermaid
flowchart BT

userInputId([User])

subgraph agentId[Agent]
    reasoningGenerationId[Reasoning Generation]
    actionGeneratorId[Action Generation]
    memoryId[Memory]
end

subgraph envId[Environment]
    toolId1[Tool 1]
    toolId2[Tool 2]
    ...
end

userInputId -->|User query| reasoningGenerationId
actionGeneratorId -->|Final answer| userInputId

reasoningGenerationId -.->|Reasoning| actionGeneratorId
actionGeneratorId -.->|Tool calls| envId
envId -.->|Tool results| memoryId
memoryId -.->|History of toolcalls & reasonings| actionGeneratorId
```

## Quick Start

```bash
uv sync --extra openai

uv run examples/agents/react/main.py
```

---

## Concept

The `TinyReActAgent` emits several **[hooks](../hooks/README.md)** during execution.  
You can subclass the agent and override these methods, or attach callbacks, to handle custom logging, monitoring, or UI integration.

| Hook                          | Trigger                                                             |
|-------------------------------|---------------------------------------------------------------------|
| `on_answer(answer: str)`      | When the agent emits a **final answer**.                            |
| `on_tool_reasoning(text: str)`| When the **ReasoningTool** is used and produces intermediate reasoning. |
| `on_error(error: Exception)`  | When an **exception** occurs during reasoning or tool execution.    |

---

## Hooks

`TinyReActAgent` inherits the hook surface from `TinyBaseAgent` and invokes these callbacks during a run:

| Hook | Trigger |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `on_before_llm_call(*, run_id, llm_input)` | Fired before every LLM call (reasoning, action orchestration, and fallback prompts). |
| `on_after_llm_call(*, run_id, llm_input, result)` | Runs after each LLM call completes; streaming calls finish with `result=None` once all chunks arrive. |
| `on_before_tool_call(*, run_id, tool, args)` | Fired immediately before any tool is executed. |
| `on_after_tool_call(*, run_id, tool, args, result)` | Fired after a tool executes successfully, including the tool output. |
| `on_tool_reasoning(*, run_id, reasoning)` | Emitted whenever a `ReasoningTool` shares intermediate reasoning text. |
| `on_answer_chunk(*, run_id, chunk, idx)` | Emitted for every streamed chunk returned by `run_stream`. |
| `on_answer(*, run_id, answer)` | Emitted once the blocking `run` method aggregates and returns the final answer. |
| `on_error(*, run_id, e)` | Triggered whenever reasoning, tool execution, or streaming raises an exception. |

---

## Files

* `example.py` — runnable demo with two example tools.
* `agent.yaml` — prompt templates for reasoning and action generation.

---

## Quick Run

```bash
tiny \
    -i examples/agents/react/main.py \
    terminal \
    -c examples/agents/react/agent.yaml \
    -q "What is the best travel destination and what is the weather like there?"
```

---

## Example Tools

```python
from tinygent.tools import tool
from tinygent.core.types import TinyModel
from pydantic import Field

class WeatherInput(TinyModel):
    location: str = Field(..., description="The location to get the weather for.")

@tool
def get_weather(data: WeatherInput) -> str:
    """Get the current weather in a given location."""
    return f"The weather in {data.location} is sunny with a high of 75°F."


class GetBestDestinationInput(TinyModel):
    top_k: int = Field(..., description="The number of top destinations to return.")

@tool
def get_best_destination(data: GetBestDestinationInput) -> list[str]:
    """Get the best travel destinations."""
    destinations = ["Paris", "New York", "Tokyo", "Barcelona", "Rome"]
    return destinations[: data.top_k]
```

---

## Example Agent

```python
from pathlib import Path
from tinygent.agents import TinyReActAgent
from tinygent.agents.react_agent import (
    ActionPromptTemplate,
    ReasonPromptTemplate,
    ReActPromptTemplate,
)
from tinygent.llms import OpenAILLM
from tinygent.utils import tiny_yaml_load

react_agent_prompt = tiny_yaml_load(str(Path(__file__).parent / 'prompts.yaml'))

react_agent = TinyReActAgent(
    llm=OpenAILLM(),
    prompt_template=ReActPromptTemplate(
        reason=ReasonPromptTemplate(
            init=react_agent_prompt['reason']['init'],
            update=react_agent_prompt['reason']['update'],
        ),
        action=ActionPromptTemplate(action=react_agent_prompt['action']['action']),
    ),
    tools=[get_weather, get_best_destination],
)
```

---

## Running the Agent

### Blocking Mode

```python
result = react_agent.run(
    "Find the best travel destination and tell me the weather there."
)
print("[RESULT]", result)
print("[MEMORY]", react_agent.memory.load_variables())
```

### Streaming Mode

Use `run_stream` for incremental reasoning/tool updates suitable for live UIs or logs:

```python
import asyncio

async def stream_demo():
    async for chunk in react_agent.run_stream(
        "Find the best travel destination and tell me the weather there."
    ):
        print("[STREAM CHUNK]", chunk)

asyncio.run(stream_demo())
```

---

## Expected Output

```
[USER INPUT] Find the best travel destination and tell me the weather there.
--- ITERATION 1 ---
[1. ITERATION - Reasoning]: I should first decide the best destination.
[1. ITERATION - Tool Call]: get_best_destination({'top_k': 1}) = ['Tokyo']
--- ITERATION 2 ---
[2. ITERATION - Reasoning]: Now I should check the weather in Tokyo.
[2. ITERATION - Tool Call]: get_weather({'location': 'Tokyo'}) = The weather in Tokyo is sunny with a high of 75°F.
[RESULT] The best travel destination is Tokyo, and the weather there is sunny with a high of 75°F.
[MEMORY] {'chat_history': '... full conversation log ...'}
```
