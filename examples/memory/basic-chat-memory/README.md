### BufferChatMemory — Conversation Buffer

`BufferChatMemory` is the simplest memory backend for `tinygent`.
It keeps an in-memory buffer of all messages exchanged in a conversation.

* **Concept**: every message (user or agent) is appended to an internal `BaseChatHistory` object.

* **Available methods**:
  * `save_context(message)`: store a new message in the buffer.
  * `load_variables()`: expose the current buffer as a single string, ready to be injected into the next prompt.
  * `clear()`: drop the buffer entirely and start fresh.

* **How it works in practice**:
  1. Memory starts empty.
  2. Each call to `save_context` appends a new message.
  3. `load_variables` lets you pass the full conversation back into the model for continuity.
  4. When you no longer need past context, `clear` resets everything.

This is a simple, in-process memory — nothing is persisted to disk or external storage. It’s designed for examples, prototypes, or any agent that only needs short-term conversational context.
