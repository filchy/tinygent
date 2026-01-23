# BufferWindowChatMemory — Sliding Window Conversation Memory

`BufferWindowChatMemory` is a variant of `BufferChatMemory` that only retains the *last `k` messages* in a conversation.
This is useful when you want to keep the most recent context short and focused, instead of passing the entire conversation back to the model.

---

## Concept

* Stores messages in a `BaseChatHistory`.
* When reading memory (`load_variables` or `chat_buffer_window`), it returns only the most recent `k` messages.
* Perfect for keeping prompts lightweight while still giving the model near-term context.

---

## API

* `k`: integer (default `5`). How many messages to retain.
* `save_context(message)`: add a new message to memory.
* `chat_buffer_window()`: returns a list of the last `k` messages.
* `load_variables()`: expose the windowed buffer as a string, ready to inject into the next prompt.
* `clear()`: reset all stored messages.

---

## Example

```python
from tinygent.core.datamodels.messages import TinyHumanMessage, TinyChatMessage
from tinygent.memory import BufferWindowChatMemory

memory = BufferWindowChatMemory()
memory.k = 3  # keep only the last 3 messages

memory.save_context(TinyHumanMessage(content="Hello"))
memory.save_context(TinyChatMessage(content="Hi there!"))
memory.save_context(TinyHumanMessage(content="What's up?"))
memory.save_context(TinyChatMessage(content="All good!"))
memory.save_context(TinyHumanMessage(content="Thanks!"))

print("Windowed memory:", memory.chat_buffer_window())
print("As variables:", memory.load_variables())
```

**Output:**

```
Windowed memory: [TinyChatMessage("All good!"), TinyHumanMessage("Thanks!")]
As variables: {'last_3_messages': '...'}
```

---

## When to Use

* Keep prompts short and within token limits.
* Provide only *recent context* to the model.
* Ideal for long conversations where full history is unnecessary.
