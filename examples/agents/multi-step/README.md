# TinyMultiStepAgent Example — Multi-Step Reason + Act Agent

This example demonstrates how to build and run a **multi-step ReAct-style agent** (`TinyMultiStepAgent`) using `tinygent`.
The agent alternates between **planning** and **acting**, while keeping track of steps, tools, and conversation history.

---

## Concept

* **Planning**: the agent generates or updates a plan every `plan_interval` turns (default: 5).
* **Acting**: the agent executes planned actions step by step, calling tools when needed.
* **Final Answer**: if no final answer is reached within `max_steps` (default: 15), the agent generates one explicitly.
* **Memory**: stores conversation history using `BufferChatMemory` (or any other memory backend).
* **Tools**: user-defined functions decorated with `@tool`.

---

## Files

* `example.py` — runnable demo with two example tools.
* `agent.yaml` — prompt templates for planning, acting, and final answer generation.

---

## Example Tools

```python
from tinygent.tools.tool import tool
from tinygent.types.base import TinyModel
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
from tinygent.agents.multi_step_agent import (
    ActionPromptTemplate,
    FinalAnswerPromptTemplate,
    PlanPromptTemplate,
    MultiStepPromptTemplate,
    TinyMultiStepAgent,
)
from tinygent.llms import OpenAILLM
from tinygent.memory import BufferChatMemory
from tinygent.utils.load_file import load_yaml

multi_step_agent_prompt = load_yaml(str(Path(__file__).parent / "agent.yaml"))

multi_step_agent = TinyMultiStepAgent(
    llm=OpenAILLM(),
    memory_list=[BufferChatMemory()],
    prompt_template=MultiStepPromptTemplate(
        acter=ActionPromptTemplate(
            system=multi_step_agent_prompt["acter"]["system"],
            final_answer=multi_step_agent_prompt["acter"]["final_answer"],
        ),
        plan=PlanPromptTemplate(
            init_plan=multi_step_agent_prompt["planner"]["init_plan"],
            update_plan=multi_step_agent_prompt["planner"]["update_plan"],
        ),
        final=FinalAnswerPromptTemplate(
            final_answer=multi_step_agent_prompt["final"]["final_answer"],
        ),
    ),
    tools=[get_weather, get_best_destination],
)
```

---

## Running the Agent

```python
result = multi_step_agent.run(
    "What is the best travel destination and what is the weather like there?"
)
print("[RESULT]", result)
print("[MEMORY]", multi_step_agent.memory.load_variables())
```

---

## Expected Output

```
[USER INPUT] What is the best travel destination and what is the weather like there?
--- STEP 1 ---
[1. STEP - Plan]: Decide how to pick destination and check weather.
[1. STEP - Tool Call]: get_best_destination({'top_k': 1}) = ['Paris']
[2. STEP - Tool Call]: get_weather({'location': 'Paris'}) = The weather in Paris is sunny with a high of 75°F.
[RESULT] The best destination is Paris. The weather in Paris is sunny with a high of 75°F.
[MEMORY] {'chat_history': '... full conversation log ...'}
```

---

## When to Use

* You want an agent that **reasons over multiple steps** before answering.
* You need **plans that can be updated** during execution.
* You want **tool use** deeply integrated into reasoning and final answers.
* You want **memory persistence** for conversation continuity.

This setup is ideal for complex workflows (e.g., travel planning, research assistants, or multi-tool problem solving) where a **plan → act → revise** loop improves results.
