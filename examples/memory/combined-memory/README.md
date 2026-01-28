# CombinedMemory — Compose Multiple Memory Backends

`CombinedMemory` lets you wire together multiple memory strategies and expose them as a single memory to an agent. Each underlying memory receives all messages; their exported variables are merged. This is useful when you want both a full transcript and a short sliding window (or any other custom combination) simultaneously.

---

## Concept

Internally, `CombinedMemory` holds a list of concrete memory instances (e.g. `BufferChatMemory`, `BufferWindowChatMemory`, custom implementations). When you call:

* `save_context(message)` — every underlying memory stores the message.
* `load_variables()` / `aload_variables()` — it merges the dicts returned by each child memory (later keys override earlier ones on collision).
* `clear()` — clears all memories.

Because each memory can expose different variable names, you can design prompt templates that selectively use them.

---

## API

```python
CombinedMemory(memory_list=[MemoryA(), MemoryB(), ...])

memory.memory_keys        # union of all child memory keys
memory.save_context(msg)  # broadcasts to all
memory.load_variables()   # merged dict
memory.clear()            # clears all
```

Async variants (`asave_context`, `aload_variables`, `aclear`) are also available; they run the underlying calls concurrently.

---

## Example

```python
from tinygent.core.datamodels.messages import TinyHumanMessage, TinyChatMessage
from tinygent.memory import BufferChatMemory, BufferWindowChatMemory
from tinygent.memory import CombinedMemory

combined = CombinedMemory(memory_list=[
	BufferChatMemory(),                # full transcript
	BufferWindowChatMemory(k=3),       # recent window
])

combined.save_context(TinyHumanMessage(content="Hello"))
combined.save_context(TinyChatMessage(content="Hi!"))
combined.save_context(TinyHumanMessage(content="Give me a plan."))
combined.save_context(TinyChatMessage(content="1. Code\n2. Walk\n3. Read"))
combined.save_context(TinyHumanMessage(content="Thanks"))

print("Memory keys:", combined.memory_keys)
print("Merged variables:\n", combined.load_variables())
print("Combined __str__:\n", combined)
```

---

## When to Use

Use `CombinedMemory` when you need multiple views of conversation state at once: e.g. full history for long-term reasoning plus a short window for concise prompts, or mixing structured summaries with raw messages.

---

## Notes

* Key collisions: later memories in the `memory_list` overwrite earlier keys when merging.
* Order matters if you depend on override behavior; place specific/shorter memories later.
* Keep an eye on total token contribution if you merge large buffers.
