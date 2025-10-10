# Example: LLM Function Calling with TinyGent Tools

This example shows how to:

* Declare simple tools with `@tool` (automatically registered into the global registry)
* Ask an LLM to decide when and which tool to call
* Iterate model output (chat vs. tool calls) with a helper iterator
* Execute the selected tool by name with validated arguments (`TinyModel`, dict, or kwargs)

This is an example, not formal API documentation.

---

## 1) Define a couple of tools

Each tool accepts a single Pydantic model as input. The `@tool` decorator wraps and registers the function under its Python name (`get_weather`, `get_time`).

```python
from langchain_core.prompt_values import StringPromptValue
from pydantic import TinyModel, Field

from tinygent.llms import OpenAILLM
from tinygent.runtime.global_registry import GlobalRegistry
from tinygent.tools.tool import tool


class GetWeatherInput(TinyModel):
    location: str = Field(..., description='The location to get the weather for.')


@tool
def get_weather(data: GetWeatherInput) -> str:
    """Get the current weather in a given location."""
    return f"The weather in {data.location} is sunny with a high of 75°F."


class GetTimeInput(TinyModel):
    location: str = Field(..., description='The location to get the time for.')


@tool
def get_time(data: GetTimeInput) -> str:
    """Get the current time in a given location."""
    return f"The current time in {data.location} is 2:00 PM."
```

Every tool defined with `@tool` is automatically registered in the global registry. You can retrieve it later with:

```python
GlobalRegistry.get_registry().get_tool("get_weather")
```

---

## 2) Minimal LLM call that enables tools

Ask the model a question and provide a list of tools it can choose from. The model may return plain chat content or one or more tool calls.

```python
if __name__ == '__main__':
    my_tools = [get_weather, get_time]

    openai_llm = OpenAILLM()

    response = openai_llm.generate_with_tools(
        prompt=StringPromptValue(text='What is the weather like in New York?'),
        tools=my_tools,
    )

    for message in response.tiny_iter():
        if message.type == 'chat':
            print(f'LLM response: {message.content}')

        elif message.type == 'tool':
            selected_tool = GlobalRegistry.get_registry().get_tool(message.tool_name)
            result = selected_tool(**message.arguments)
            print('Tool %s called with arguments %s, result: %s' % (
                message.tool_name,
                message.arguments,
                result,
            ))
```

Calling `selected_tool(**message.arguments)` works because the `Tool` wrapper validates dict/kwargs against the tool’s Pydantic schema and dispatches to the underlying function.

---

## 3) Tiny iterator for mixed LLM outputs

The example uses a helper to iterate over the model’s mixed output (chat or tool call). This allows you to process tool calls in the same loop as plain LLM messages.

```python
from itertools import chain
from typing import Iterator, Literal, cast
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tinygent.types.base import TinyModel
from tinygent.llms.utils import normalize_content


class TinyChatMessage(TinyModel):
    type: Literal['chat'] = 'chat'
    content: str
    metadata: dict = {}


class TinyToolCall(TinyModel):
    type: Literal['tool'] = 'tool'
    tool_name: str
    arguments: dict
    call_id: str | None = None
    metadata: dict = {}


TinyMessageLiteral = Literal['chat', 'tool']
TinyAIMessage = TinyChatMessage | TinyToolCall


class TinyLLMResult(LLMResult):
    def tiny_iter(self) -> Iterator[TinyAIMessage]:
        for generation in chain.from_iterable(self.generations):
            chat_gen = cast(ChatGeneration, generation)
            message = chat_gen.message

            if not isinstance(message, AIMessage):
                raise ValueError('Unsupported message type %s' % type(message))

            if (tool_calls := message.tool_calls):
                for tool_call in tool_calls:
                    yield TinyToolCall(
                        tool_name=tool_call['name'],
                        arguments=tool_call['args'],
                        call_id=tool_call['id'] or None,
                        metadata={'raw': tool_call}
                    )
            elif (content := message.content):
                yield TinyChatMessage(
                    content=normalize_content(content),
                    metadata={'raw': message}
                )
```

---

## 4) Abstract LLM interface

For flexibility, TinyGent defines an abstract base class that all LLM wrappers follow. This ensures consistent APIs across providers.

Not all LLMs support function/tool calling. Before invoking `generate_with_tools`, check the `supports_tool_calls` property of the LLM instance. If it is `False`, you should fall back to plain text generation.

```python
class AbstractLLM(ABC, Generic[LLMConfigT]):
    @property
    def supports_tool_calls(self) -> bool: ...

    @abstractmethod
    def generate_with_tools(self, prompt: PromptValue, tools: list[Tool]) -> TinyLLMResult: ...

    @abstractmethod
    async def agenerate_with_tools(self, prompt: PromptValue, tools: list[Tool]) -> TinyLLMResult: ...
```

Only the relevant parts of the abstract base class are shown here. In practice, it also includes methods for plain text generation and structured outputs.

---

## Running the example

```bash
uv run main.py
```

Expected output (weather wording may vary):

```
Tool get_weather called with arguments {'location': 'New York'}, result: The weather in New York is sunny with a high of 75°F.
```
