# Memory

Memory allows agents to remember conversation history and context across multiple interactions.

---

## What is Memory?

Without memory, each agent call is independent:

```python
agent = build_agent('react', llm='openai:gpt-4o-mini', tools=[...])

agent.run("My name is Alice")
# Agent: "Nice to meet you, Alice!"

agent.run("What is my name?")
# Agent: "I don't know your name."  No memory
```

With memory, agents remember context:

```python
from tinygent.memory import BufferChatMemory

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[...],
    memory=BufferChatMemory()
)

agent.run("My name is Alice")
# Agent: "Nice to meet you, Alice!"

agent.run("What is my name?")
# Agent: "Your name is Alice!"  Remembers
```

---

## Memory Types

Tinygent provides 4 built-in memory types:

### 1. BufferChatMemory

**Best for**: Short conversations, full history needed

Stores all messages in a list:

```python
from tinygent.memory import BufferChatMemory

memory = BufferChatMemory()

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[...],
    memory=memory
)

agent.run("Hello")
agent.run("My name is Bob")
agent.run("What's my name?")

# View history
print(memory.load_variables())
# [
#   HumanMessage("Hello"),
#   AIMessage("Hi there!"),
#   HumanMessage("My name is Bob"),
#   AIMessage("Nice to meet you, Bob!"),
#   HumanMessage("What's my name?"),
# ]
```

**Pros:**

- Simple and reliable
- Complete conversation history
- No information loss

**Cons:**

- Grows unbounded
- Can exceed token limits
- Expensive for long conversations

---

### 2. SummaryBufferMemory

**Best for**: Long conversations, summarization acceptable

Summarizes old messages to save tokens:

```python
from tinygent.memory import SummaryBufferMemory

memory = SummaryBufferMemory(
    llm=build_llm('openai:gpt-4o-mini'),
    max_token_limit=500,  # Summarize when exceeded
)

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[...],
    memory=memory
)

# After many messages, old ones get summarized
agent.run("Tell me about AI")  # 200 tokens
agent.run("What about ML?")    # 200 tokens
agent.run("And deep learning?") # 200 tokens
# Now at 600 tokens → triggers summary

# Old messages condensed to summary
print(memory.load_variables())
# [
#   SystemMessage("Summary: User asked about AI and ML..."),
#   HumanMessage("And deep learning?"),
#   AIMessage("Deep learning is..."),
# ]
```

**Pros:**

- Handles long conversations
- Prevents token limit issues
- Maintains key information

**Cons:**

- Loses details in summary
- Extra LLM calls for summarization
- May miss nuances

---

### 3. WindowBufferMemory

**Best for**: Recent context only, sliding window

Keeps only the last N messages:

```python
from tinygent.memory import WindowBufferMemory

memory = WindowBufferMemory(window_size=4)  # Keep last 4 messages

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[...],
    memory=memory
)

agent.run("Message 1")
agent.run("Message 2")
agent.run("Message 3")
agent.run("Message 4")

# Window is full: [User1, AI1, User2, AI2]

agent.run("Message 5")

# Oldest message dropped: [User2, AI2, User3, AI3]
```

**Pros:**

- Predictable memory usage
- Fast and simple
- Good for recent context

**Cons:**

- Forgets old information
- No long-term memory
- May lose important context

---

### 4. CombinedMemory

**Best for**: Multiple memory strategies simultaneously

Combine different memory types:

```python
from tinygent.memory import CombinedMemory
from tinygent.memory import BufferChatMemory
from tinygent.memory import WindowBufferMemory

# Full history + recent window
combined = CombinedMemory(
    memories={
        'full_history': BufferChatMemory(),
        'recent': WindowBufferMemory(window_size=6),
    }
)

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[...],
    memory=combined
)

# Both memories updated simultaneously
agent.run("Important information from the start")
# ... many messages ...
agent.run("Recent question")

# Access specific memory
full = combined.memories['full_history'].load_variables()
recent = combined.memories['recent'].load_variables()
```

**Pros:**

- Flexible combinations
- Multiple access patterns
- Customizable strategies

**Cons:**

- More complex setup
- Higher memory usage

---

## Memory Operations

### Saving Context

Manually save messages:

```python
from tinygent.core.datamodels.messages import TinyHumanMessage, TinyChatMessage

memory = BufferChatMemory()

# Save user message
user_msg = TinyHumanMessage(content="Hello")
memory.save_context(user_msg)

# Save AI response
ai_msg = TinyChatMessage(content="Hi there!")
memory.save_context(ai_msg)
```

### Loading Variables

Retrieve conversation history:

```python
# Get all messages
messages = memory.load_variables()

for msg in messages:
    print(f"{msg.role}: {msg.content}")
# human: Hello
# assistant: Hi there!
```

### Clearing Memory

Reset conversation:

```python
memory.clear()

# Memory is now empty
print(memory.load_variables())  # []
```

---

## Message Types

Tinygent supports multiple message types:

```python
from tinygent.core.datamodels.messages import (
    TinyHumanMessage,      # User messages
    TinyChatMessage,       # AI responses
    TinySystemMessage,     # System prompts
    TinyPlanMessage,       # Planning messages
    TinyToolMessage,       # Tool results
)

memory = BufferChatMemory()

memory.save_context(TinySystemMessage(content="You are a helpful assistant"))
memory.save_context(TinyHumanMessage(content="Hello"))
memory.save_context(TinyChatMessage(content="Hi there!"))
memory.save_context(TinyPlanMessage(content="Plan: 1. Greet user 2. Ask how to help"))
```

---

## Memory Filtering

Filter messages by type:

```python
from tinygent.core.datamodels.messages import TinyHumanMessage, TinyChatMessage

memory = BufferChatMemory()

# Add various messages
memory.save_context(TinyHumanMessage(content="User message 1"))
memory.save_context(TinyChatMessage(content="AI response 1"))
memory.save_context(TinyHumanMessage(content="User message 2"))
memory.save_context(TinyChatMessage(content="AI response 2"))

# Add filter: only human messages
memory._chat_history.add_filter(
    'only_human',
    lambda m: isinstance(m, TinyHumanMessage)
)

print(memory._chat_history)
# Only shows:
# - User message 1
# - User message 2

# Remove filter
memory._chat_history.remove_filter('only_human')
```

---

## Advanced Patterns

### Custom Memory

Create custom memory classes:

```python
from tinygent.memory import BaseMemory

class KeywordMemory(BaseMemory):
    """Memory that only saves messages containing keywords."""

    def __init__(self, keywords: list[str]):
        super().__init__()
        self.keywords = keywords
        self.messages = []

    def save_context(self, message):
        # Only save if contains keyword
        if any(kw in message.content for kw in self.keywords):
            self.messages.append(message)

    def load_variables(self):
        return self.messages

    def clear(self):
        self.messages = []

# Use it
memory = KeywordMemory(keywords=['important', 'urgent', 'critical'])

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[...],
    memory=memory
)

agent.run("This is important information")  # Saved
agent.run("Just casual chat")                # Not saved
agent.run("Urgent: respond ASAP")           # Saved
```

### Persistent Memory

Save memory to disk:

```python
import json
from pathlib import Path

def save_memory(memory, filepath: str):
    """Save memory to JSON file."""
    messages = [
        {'role': msg.role, 'content': msg.content}
        for msg in memory.load_variables()
    ]
    Path(filepath).write_text(json.dumps(messages, indent=2))

def load_memory(filepath: str) -> BufferChatMemory:
    """Load memory from JSON file."""
    memory = BufferChatMemory()
    messages = json.loads(Path(filepath).read_text())

    for msg in messages:
        if msg['role'] == 'human':
            memory.save_context(TinyHumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            memory.save_context(TinyChatMessage(content=msg['content']))

    return memory

# Usage
memory = BufferChatMemory()
agent = build_agent('react', llm='openai:gpt-4o-mini', memory=memory)

agent.run("Remember this")
save_memory(memory, 'conversation.json')

# Later...
memory = load_memory('conversation.json')
agent = build_agent('react', llm='openai:gpt-4o-mini', memory=memory)
agent.run("What did I say earlier?")  # Remembers from disk
```

---

## Memory with MultiStep Agent

MultiStep agents benefit from memory:

```python
from tinygent.agents.multi_step_agent import TinyMultiStepAgent
from tinygent.memory import BufferChatMemory

agent = TinyMultiStepAgent(
    llm=build_llm('openai:gpt-4o'),
    tools=[...],
    memory=BufferChatMemory(),
)

# First task
agent.run("Plan a trip to Prague")
# Agent creates plan, executes steps, remembers results

# Second task - can reference previous context
agent.run("Update the plan based on weather")
# Agent remembers previous plan and updates it
```

---

## Choosing the Right Memory

| Use Case | Memory Type | Why |
|----------|-------------|-----|
| Chatbot (short sessions) | BufferChatMemory | Full history, simple |
| Long conversations | SummaryBufferMemory | Prevents token overflow |
| Recent context only | WindowBufferMemory | Fast, bounded |
| Complex workflows | CombinedMemory | Multiple strategies |
| Debugging | BufferChatMemory | Full visibility |
| Production chatbot | SummaryBufferMemory | Scalable |

---

## Best Practices

### 1. Clear Memory When Needed

```python
# Start fresh conversation
if user_says_reset:
    memory.clear()
```

### 2. Monitor Memory Size

```python
messages = memory.load_variables()
if len(messages) > 50:
    print("Warning: Memory getting large")
```

### 3. Use Summaries for Long Chats

```python
# For customer support (long sessions)
memory = SummaryBufferMemory(
    llm=build_llm('openai:gpt-4o-mini'),
    max_token_limit=1000,
)
```

### 4. Window for Short Context

```python
# For quick Q&A (no long-term memory needed)
memory = WindowBufferMemory(window_size=4)
```

---

## Memory and Middleware

Track memory changes with middleware:

```python
from tinygent.agents.middleware import TinyBaseMiddleware

class MemoryMonitorMiddleware(TinyBaseMiddleware):
    def on_answer(self, *, run_id: str, answer: str) -> None:
        # Check memory size after each answer
        size = len(str(agent.memory.load_variables()))
        print(f"Memory size: {size} characters")

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[...],
    memory=BufferChatMemory(),
    middleware=[MemoryMonitorMiddleware()]
)
```

---

## Next Steps

- **[Agents](agents.md)**: Use memory with agents
- **[Middleware](middleware.md)**: Monitor memory with middleware
- **[Examples](../examples.md)**: See memory examples

---

## Examples

Check out:

- `examples/memory/basic-chat-memory/main.py` - Buffer memory
- `examples/memory/buffer-summary-memory/main.py` - Summary memory
- `examples/memory/buffer-window-chat-memory/main.py` - Window memory
- `examples/memory/combined-memory/main.py` - Combined memory
