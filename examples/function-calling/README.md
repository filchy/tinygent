# TinyGent Tools Example

This example shows how to:

* Declare simple tools with `@tool`
* Ask an LLM to decide when and which tool to call
* Iterate model output (chat vs. tool calls) with a helper iterator
* Execute the selected tool by name with validated arguments (`TinyModel`, dict, or kwargs)

This is an example, not formal API documentation.

## Quick Start

```bash
uv sync --extra openai

uv run examples/function-calling/main.py
```

---

## 1) Define a couple of tools

TinyGent supports **two ways** to define tool parameters. The `@tool` decorator wraps the function and returns a `Tool` instance. These are not automatically registered globally — you explicitly pass them to the LLM when needed.

### Variant 1: TinyModel Descriptor (Explicit Schema)

Pass a single `TinyModel` subclass for full control over field descriptions:

```python
from pydantic import Field
from tinygent.core.types import TinyModel
from tinygent.tools import tool


class GetWeatherInput(TinyModel):
    location: str = Field(..., description='The location to get the weather for.')


@tool
def get_weather(data: GetWeatherInput) -> str:
    """Get the current weather in a given location."""
    return f"The weather in {data.location} is sunny with a high of 75°F."
```

### Variant 2: Regular Parameters (Auto-Generated Schema)

Pass parameters directly like any normal function — TinyGent auto-generates the schema:

```python
from tinygent.tools import tool


@tool
def get_time(location: str) -> str:
    """Get the current time in a given location."""
    return f"The current time in {location} is 2:00 PM."
```

Both variants work identically with LLM function calling — the schema is generated automatically.

---

## 2) Minimal LLM call that enables tools

Ask the model a question and provide a list of tools it can choose from. The model may return plain chat content or one or more tool calls.

```python
from tinygent.core.types import TinyLLMInput
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.llms import OpenAILLM

if __name__ == '__main__':
    my_tools = [get_weather, get_time]

    openai_llm = OpenAILLM()

    response = openai_llm.generate_with_tools(
        llm_input=TinyLLMInput(
            messages=[TinyHumanMessage(content='What is the weather like in New York?')]
        ),
        tools=my_tools,
    )

    tool_map = {tool.info.name: tool for tool in my_tools}

    for message in response.tiny_iter():
        if message.type == 'chat':
            print(f'LLM response: {message.content}')

        elif message.type == 'tool':
            result = tool_map[message.tool_name](**message.arguments)
            print(
                'Tool %s called with arguments %s, result: %s'
                % (message.tool_name, message.arguments, result)
            )
```

---

## 3) Iterator for mixed LLM outputs

The `tiny_iter()` helper yields either `chat` or `tool` messages, so you can handle them in one loop.

---

## 4) Notes

* `Tool` validates dict/kwargs against its Pydantic schema automatically.
* Both sync and async tools are supported, including caching.
* `@tool` returns a local instance — you manage which tools to pass to the LLM explicitly.

---

## Running the example

```bash
uv run main.py
```

Expected output (weather wording may vary):

```
Tool get_weather called with arguments {'location': 'New York'}, result: The weather in New York is sunny with a high of 75°F.
```
