# TinyReActAgent Example — Reason + Act Agent

This example demonstrates how to build and run a **ReAct-style agent** (`TinyReActAgent`) using `tinygent`. The agent alternates between **planning** and **acting**, while leveraging memory and tools.

---

## Concept

* **Planning**: the agent generates a plan of steps every `plan_interval` turns (default: 5).
* **Acting**: the agent executes actions step by step, calling tools when needed.
* **Memory**: stores conversation history using `BufferChatMemory` (or any other memory backend).
* **Tools**: user-defined functions decorated with `@tool`.

---

## Files

* `example.py` — runnable demo with two example tools.
* `react_agent.yaml` — prompt templates for planning and acting.

---

## Example Tools

```python
from tinygent.tools.tool import tool
from tinygent.types import TinyModel
from pydantic import Field

class WeatherInput(TinyModel):
    location: str = Field(..., description="The location to get the weather for.")

@tool
def get_weather(data: WeatherInput) -> str:
    return f"The weather in {data.location} is sunny with a high of 75°F."

class GetBestDestinationInput(TinyModel):
    top_k: int = Field(..., description="The number of top destinations to return.")

@tool
def get_best_destination(data: GetBestDestinationInput) -> list[str]:
    destinations = ["Paris", "New York", "Tokyo", "Barcelona", "Rome"]
    return destinations[: data.top_k]
```

---

## Example Agent

```python
from pathlib import Path
from tinygent.agents.react_agent import TinyReActAgent, PlanPromptTemplate, ActionPromptTemplate, ReactPromptTemplate
from tinygent.llms.openai import OpenAILLM
from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.utils.load_file import load_yaml

react_agent_prompt = load_yaml(str(Path(__file__).parent / "react_agent.yaml"))

react_agent = TinyReActAgent(
    llm=OpenAILLM(),
    memory_list=[BufferChatMemory()],
    prompt_template=ReactPromptTemplate(
        acter=ActionPromptTemplate(
            system=react_agent_prompt["acter"]["system"],
            final_answer=react_agent_prompt["acter"]["final_answer"],
        ),
        plan=PlanPromptTemplate(
            init_plan=react_agent_prompt["planner"]["init_plan"],
            update_plan=react_agent_prompt["planner"]["update_plan"],
        ),
    ),
    tools=[get_weather, get_best_destination],
)
```

---

## Running the Agent

```python
result = react_agent.run(
    "What is the best travel destination and what is the weather like there?"
)
print("[RESULT]", result)
print("[MEMORY]", react_agent.memory.load_variables())
```

---

## Expected Output

```
[USER INPUT] What is the best travel destination and what is the weather like there?
--- STEP 1 ---
[PLAN] Decide how to pick destination and check weather.
[AGENT OUTPUT] ToolCall: get_best_destination
[AGENT OUTPUT] ToolCall: get_weather
[RESULT] The best destination is Paris. The weather in Paris is sunny with a high of 75°F.
[MEMORY] {'chat_history': '... full conversation log ...'}
```

---

## When to Use

* You want an agent that **reasons about tasks** before acting.
* You need **tool use** integrated into reasoning.
* You want **memory persistence** for conversation continuity.

This setup is ideal for travel planning, research assistants, or any domain where a plan → act loop improves results.
