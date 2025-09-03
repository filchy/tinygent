### BaseChatMemory — Simple Conversation Memory

`BaseChatMemory` provides a lightweight way to keep track of interactions during a conversation.

* **Core idea**: every user input and model output is stored as a message inside an internal chat history.

* **Main actions**:

  * `save_context(...)`: adds the latest input and output to the history.
  * `load_variables()`: returns the current history as a string, so you can pass it into the next prompt if you want continuity.
  * `clear()`: resets the history to empty.

* **Typical flow**:

  1. Start with empty memory.
  2. Send a prompt → model replies → both get stored.
  3. Ask a follow-up → memory still contains the previous exchange, which you can feed back in.
  4. Call `clear()` when you want to wipe the slate clean.

This is a simple, in-process memory — nothing is persisted to disk or external storage. It’s designed for examples, prototypes, or any agent that only needs short-term conversational context.
