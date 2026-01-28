````markdown
# BufferSummaryChatMemory — Summarizing Conversation Memory

`BufferSummaryChatMemory` is a memory backend that keeps recent messages in a buffer and automatically summarizes older messages when the buffer exceeds a token limit. This gives the model both precise recent context and a compressed summary of earlier conversation.

---

## Concept

* Stores messages in a `BaseChatHistory`.
* When the buffer exceeds `max_token_limit`, it prunes the oldest messages and uses an LLM to generate/update a running summary.
* The summary is stored as a `TinySummaryMessage` and prepended to the buffer when loading variables.
* Perfect for long conversations where you need full context awareness without exceeding token limits.

---

## API

* `llm`: the LLM instance used to generate summaries.
* `max_token_limit`: integer (default `2000`). Maximum tokens before pruning and summarizing.
* `return_messages`: boolean (default `False`). If `True`, returns a list of messages; otherwise returns a formatted string.
* `save_context(message)`: add a new message and trigger pruning if needed.
* `load_variables()`: returns the summary (if any) followed by the recent buffer.
* `prune()`: manually trigger the summarization of older messages.
* `clear()`: reset all stored messages and summary.

---

## Example

```python
from tinygent.core.datamodels.messages import TinyChatMessage, TinyHumanMessage, TinyPlanMessage
from tinygent.core.factory import build_llm
from tinygent.memory import BufferSummaryChatMemory

memory = BufferSummaryChatMemory(
    build_llm('openai:gpt-4o-mini'),
    max_token_limit=30,
    return_messages=True,
)

# First exchange
memory.save_context(TinyHumanMessage(content='Hello, assistant.'))
memory.save_context(TinyChatMessage(content='Hi there! How can I help you today?'))

# Second exchange
memory.save_context(TinyHumanMessage(content='Can you make a plan for my weekend?'))
memory.save_context(TinyPlanMessage(content='Sure! 1. Go hiking. 2. Watch a movie. 3. Relax.'))

print('Full memory:', memory.load_variables())
```

**Output:**

```
Full memory: {'summarized_chat': [TinySummaryMessage(type='summary', metadata={}, content='The human initiated a conversation with the assistant, requesting assistance in planning their weekend.'), TinyPlanMessage(type='plan', metadata={}, content='Sure! 1. Go hiking. 2. Watch a movie. 3. Relax.')]}
```

The oldest messages (greeting exchange and weekend planning request) are summarized into a `TinySummaryMessage`, while the most recent `TinyPlanMessage` remains in the buffer.

---

## When to Use

* Long-running conversations where full history would exceed token limits.
* When you need to preserve important context from earlier in the conversation.
* Ideal for agents that require awareness of past interactions without passing every message.

---

## Notes

* The quality of summaries depends on the LLM you provide.
* Lower `max_token_limit` values trigger more frequent summarization (useful for testing).
* The summary accumulates over time — new pruned messages are merged into the existing summary.

````
