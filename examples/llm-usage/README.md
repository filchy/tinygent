# LLM Usage Example

This example demonstrates how to use a custom LLM implementation (`OpenAILLM`) that conforms to the `AbstractLLM` interface in **tinygent**. It shows basic usage of different generation methods, including:

* Basic text generation
* Structured output generation
* Tool-enhanced function calling
* Asynchronous generation

## Quick Start

```bash
uv sync --extra openai

uv run examples/llm-usage/main.py
```

---

## Requirements

* You must implement an LLM class inheriting from `AbstractLLM`
* Your LLM must return instances of `TinyLLMResult` for all supported generation methods
* Tools must be registered using the `@tool` decorator and conform to the `Tool` interface

---

## Interface: `AbstractLLM`

The `AbstractLLM` defines the following required methods:

```python
class AbstractLLM(ABC):

    def generate_text(...) -> TinyLLMResult
    async def agenerate_text(...) -> TinyLLMResult

    def generate_structured(...) -> LLMStructuredT
    async def agenerate_structured(...) -> LLMStructuredT

    def generate_with_tools(...) -> TinyLLMResult
    async def agenerate_with_tools(...) -> TinyLLMResult
```

Each method must return a `TinyLLMResult`, which provides a `.tiny_iter()` generator that yields parsed messages:

* `TinyChatMessage` for raw LLM responses
* `TinyToolCall` for function/tool invocations

---

## Example: Basic Text Generation

```python
def basic_generation():
    llm = OpenAILLM()
    result = llm.generate_text(
        prompt=StringPromptValue(text="Tell me a joke about programmers.")
    )
    for msg in result.tiny_iter():
        print(msg.content)
```

---

## Example: Structured Output Generation

```python
class SummaryResponse(TinyModel):
    summary: str

result = llm.generate_structured(
    prompt=StringPromptValue(text="Summarize why the sky is blue."),
    output_schema=SummaryResponse
)

print(result.summary)
```

---

## Example: Generation With Tools

```python
def generation_with_tools():
    tools = [add, capitalize]
    llm = OpenAILLM()

    result = llm.generate_with_tools(
        prompt=StringPromptValue(
            text="Capitalize 'tinygent is powerful'. Then add 5 and 7."
        ),
        tools=tools
    )

    for msg in result.tiny_iter():
        if msg.type == 'chat':
            print("LLM Response:", msg.content)
        elif msg.type == 'tool':
            tool_fn = GlobalToolCatalog.get_active_catalog().get_tool(msg.tool_name)
            output = tool_fn(**msg.arguments)
            print(f"Tool Call {msg.tool_name}({msg.arguments}) => {output}")
```

---

## Example: Asynchronous Generation

```python
async def async_generation():
    llm = OpenAILLM()
    result = await llm.agenerate_text(
        prompt=StringPromptValue(text="Name three uses of AI in medicine.")
    )

    for msg in result.tiny_iter():
        print("[Async Message]", msg.content)
```

---

## Running the Example

```bash
uv run llm_usage.py
```

Expected output:

```
[BASIC TEXT GENERATION] LLM joke output...
[STRUCTURED RESULT] The sky appears blue because...
[LLM RESPONSE] ...
[TOOL CALL] capitalize(...) => ...
[TOOL CALL] add(...) => 12
[ASYNC TEXT GENERATION] ...
```

---

## Notes

* `generate_with_tools` works with both sync and async tools
* `TinyLLMResult.tiny_iter()` always yields normalized, structured outputs
* Every `@tool`-decorated function is globally registered under its function name
* The LLM implementation must handle OpenAI-style tool schemas when constructing the API call
