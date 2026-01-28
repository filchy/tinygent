# LLMs (Language Models)

Tinygent supports multiple LLM providers with a unified interface. Switch between OpenAI, Anthropic, Mistral, and Gemini without changing your code.

---

## Provider String Format

All LLMs in Tinygent use the format:

```
provider:model
```

**Examples:**

```python
# OpenAI
llm = build_llm('openai:gpt-4o')
llm = build_llm('openai:gpt-4o-mini')
llm = build_llm('openai:gpt-3.5-turbo')

# Anthropic Claude
llm = build_llm('anthropic:claude-3-5-sonnet')
llm = build_llm('anthropic:claude-3-5-haiku')
llm = build_llm('anthropic:claude-3-opus')

# Mistral AI
llm = build_llm('mistralai:mistral-large-latest')
llm = build_llm('mistralai:mistral-small-latest')

# Google Gemini
llm = build_llm('gemini:gemini-2.0-flash-exp')
llm = build_llm('gemini:gemini-pro')
```

---

## Supported Providers

### OpenAI

**Installation:**

```bash
uv sync --extra openai
```

**Environment Variable:**

```bash
export OPENAI_API_KEY="sk-..."
```

**Available Models:**

- `gpt-4o` - Latest GPT-4 Optimized
- `gpt-4o-mini` - Smaller, faster GPT-4
- `gpt-3.5-turbo` - Fast and cost-effective
- `gpt-4-turbo` - Previous generation

**Usage:**

```python
from tinygent.core.factory import build_llm

llm = build_llm('openai:gpt-4o-mini', temperature=0.7)

# Direct call
response = llm.generate("What is AI?")
print(response.content)

# With streaming
async for chunk in llm.stream("Tell me a story"):
    print(chunk, end='', flush=True)
```

---

### Anthropic Claude

**Installation:**

```bash
uv sync --extra anthropic
```

**Environment Variable:**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Available Models:**

- `claude-3-5-sonnet` - Best overall performance
- `claude-3-5-haiku` - Fast and efficient
- `claude-3-opus` - Most capable (expensive)

**Usage:**

```python
llm = build_llm('anthropic:claude-3-5-sonnet', temperature=0.5)

response = llm.generate("Explain quantum computing")
print(response.content)
```

---

### Mistral AI

**Installation:**

```bash
uv sync --extra mistralai
```

**Environment Variable:**

```bash
export MISTRAL_API_KEY="..."
```

**Available Models:**

- `mistral-large-latest` - Most capable
- `mistral-small-latest` - Fast and efficient
- `open-mistral-7b` - Open source

**Usage:**

```python
llm = build_llm('mistralai:mistral-large-latest')

response = llm.generate("What is machine learning?")
print(response.content)
```

---

### Google Gemini

**Installation:**

```bash
uv sync --extra gemini
```

**Environment Variable:**

```bash
export GEMINI_API_KEY="..."
```

**Available Models:**

- `gemini-2.0-flash-exp` - Latest Flash model
- `gemini-pro` - Production model
- `gemini-ultra` - Most capable (limited access)

**Usage:**

```python
llm = build_llm('gemini:gemini-2.0-flash-exp')

response = llm.generate("Explain neural networks")
print(response.content)
```

---

## Configuration Options

### Temperature

Controls randomness (0.0 = deterministic, 2.0 = very random):

```python
# Deterministic (good for factual tasks)
llm = build_llm('openai:gpt-4o-mini', temperature=0.0)

# Balanced (default)
llm = build_llm('openai:gpt-4o-mini', temperature=0.7)

# Creative (good for storytelling)
llm = build_llm('openai:gpt-4o-mini', temperature=1.5)
```

### Max Tokens

Limit response length:

```python
llm = build_llm('openai:gpt-4o-mini', max_tokens=500)

# Response will be truncated at ~500 tokens
response = llm.generate("Write a long essay about AI")
```

### Stop Sequences

Stop generation at specific strings:

```python
llm = build_llm(
    'openai:gpt-4o-mini',
    stop_sequences=['END', '\n\n\n', '---']
)

# Generation stops when any stop sequence is encountered
response = llm.generate("List items:\n1. ")
```

### Top P (Nucleus Sampling)

Alternative to temperature for controlling randomness:

```python
llm = build_llm('openai:gpt-4o-mini', top_p=0.9)
```

---

## Using LLMs with Agents

### Simple Agent

```python
from tinygent.core.factory import build_agent

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',  # Simple string
    tools=[...],
)
```

### Advanced Agent

```python
from tinygent.agents.react_agent import TinyReActAgent
from tinygent.core.factory import build_llm

llm = build_llm(
    'openai:gpt-4o',
    temperature=0.3,
    max_tokens=2000,
)

agent = TinyReActAgent(
    llm=llm,  # Pre-configured LLM object
    tools=[...],
)
```

---

## Direct LLM Usage

Use LLMs without agents:

### Synchronous

```python
from tinygent.core.factory import build_llm

llm = build_llm('openai:gpt-4o-mini')

# Simple generation
response = llm.generate("What is 2 + 2?")
print(response.content)  # "2 + 2 equals 4."

# With system prompt
response = llm.generate(
    "What is the weather?",
    system_prompt="You are a helpful weather assistant."
)
```

### Asynchronous

```python
import asyncio

async def main():
    llm = build_llm('openai:gpt-4o-mini')

    # Async generation
    response = await llm.agenerate("Tell me about AI")
    print(response.content)

asyncio.run(main())
```

### Streaming

```python
import asyncio

async def main():
    llm = build_llm('openai:gpt-4o-mini')

    # Stream tokens
    async for chunk in llm.stream("Write a short poem"):
        print(chunk, end='', flush=True)

asyncio.run(main())
```

---

## Function Calling

LLMs can call functions (tools):

```python
from tinygent.core.factory import build_llm
from tinygent.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny in {location}"

llm = build_llm('openai:gpt-4o-mini')

# LLM can decide to call the function
response = llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather]
)

# Response includes function call
if response.tool_calls:
    for call in response.tool_calls:
        print(f"Function: {call.name}")
        print(f"Arguments: {call.arguments}")
```

---

## Cost Comparison

Approximate costs per 1M tokens (as of 2025):

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| OpenAI | gpt-4o | $2.50 | $10.00 |
| Anthropic | claude-3-5-haiku | $0.25 | $1.25 |
| Anthropic | claude-3-5-sonnet | $3.00 | $15.00 |
| Mistral | mistral-large-latest | $2.00 | $6.00 |
| Gemini | gemini-2.0-flash-exp | $0.10 | $0.40 |

**Tip**: Use mini/haiku models for development, upgrade for production.

---

## Model Selection Guide

### For Development

- **OpenAI**: `gpt-4o-mini` - Fast, cheap, good quality
- **Anthropic**: `claude-3-5-haiku` - Fast, efficient
- **Gemini**: `gemini-2.0-flash-exp` - Cheapest option

### For Production

- **OpenAI**: `gpt-4o` - Excellent all-around
- **Anthropic**: `claude-3-5-sonnet` - Best reasoning
- **Mistral**: `mistral-large-latest` - European option

### For Complex Reasoning

- **Anthropic**: `claude-3-opus` - Most capable
- **OpenAI**: `gpt-4-turbo` - Strong reasoning

---

## Switching Providers

Tinygent makes it trivial to switch:

```python
# Try different models for the same task
models = [
    'openai:gpt-4o-mini',
    'anthropic:claude-3-5-haiku',
    'mistralai:mistral-large-latest',
    'gemini:gemini-2.0-flash-exp',
]

for model in models:
    agent = build_agent('react', llm=model, tools=[...])
    result = agent.run('What is AI?')
    print(f"{model}: {result}")
```

---

## Custom LLM Providers

Register custom LLM providers:

```python
from tinygent.core.runtime.global_registry import register_llm
from tinygent.llms.base import BaseLLM

@register_llm('custom')
class CustomLLM(BaseLLM):
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)

    async def agenerate(self, prompt: str, **kwargs):
        # Your custom implementation
        return response

# Use it
llm = build_llm('custom:my-model')
```

---

## Embeddings

For vector embeddings (RAG, semantic search):

```python
from tinygent.core.factory import build_embedder

# OpenAI embeddings
embedder = build_embedder('openai:text-embedding-3-small')

# VoyageAI embeddings
embedder = build_embedder('voyageai:voyage-2')

# Generate embeddings
vectors = embedder.embed_documents(['Hello', 'World'])
print(len(vectors[0]))  # 1536 dimensions
```

**Available Embedders:**

- `openai:text-embedding-3-small` (1536 dims)
- `openai:text-embedding-3-large` (3072 dims)
- `voyageai:voyage-2` (1024 dims)

---

## Best Practices

### 1. Use Environment Variables

```python
# Bad - Hardcoded keys
llm = build_llm('openai:gpt-4o', api_key='sk-...')

# Good - Environment variables
export OPENAI_API_KEY="sk-..."
llm = build_llm('openai:gpt-4o')
```

### 2. Start Small

```python
# Development: Use cheap models
dev_llm = build_llm('openai:gpt-4o-mini')

# Production: Upgrade when needed
prod_llm = build_llm('openai:gpt-4o')
```

### 3. Cache Results

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_llm_call(prompt: str) -> str:
    llm = build_llm('openai:gpt-4o-mini')
    return llm.generate(prompt).content

# Repeated calls use cache
result1 = cached_llm_call("What is AI?")
result2 = cached_llm_call("What is AI?")  # Instant, no API call
```

---

## Next Steps

- **[Agents](agents.md)**: Use LLMs with agents
- **[Tools](tools.md)**: Add tools to LLMs
- **[Examples](../examples.md)**: See LLM usage examples

---

## Examples

Check out:

- `examples/llm-usage/main.py` - Direct LLM usage
- `examples/function-calling/main.py` - Function calling
- `examples/embeddings/main.py` - Embeddings usage
