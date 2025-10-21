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

---

## Concept

* **Reasoning**: the agent generates reasoning steps based on the task and past context.
* **Acting**: the agent performs tool calls guided by the reasoning.
* **Iteration**: the agent loops through reasoning + acting until a final answer is produced or `max_iterations` is reached (default: 10).
* **Memory**: stores conversation history and tool results using `BufferChatMemory`.
* **Tools**: user-defined functions decorated with `@tool`.

---

## Files

* `example.py` — runnable demo with two example tools.
* `agent.yaml` — prompt templates for reasoning and action generation.

---

## Quick Run

```bash
tiny terminal \
  -c examples/agents/react/agent.yaml \
  -q "What is the weather like in Tokyo?" \
  -q "What is the best travel destination?" \
```

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
from tinygent.agents.react_agent import (
    ActionPromptTemplate,
    ReasonPromptTemplate,
    ReActPromptTemplate,
    TinyReActAgent,
)
from tinygent.llms import OpenAILLM
from tinygent.utils.yaml import tiny_yaml_load

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

```python
result = react_agent.run(
    "Find the best travel destination and tell me the weather there."
)
print("[RESULT]", result)
print("[MEMORY]", react_agent.memory.load_variables())
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
