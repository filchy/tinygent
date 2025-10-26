# Hooks Example in TinyGent

This example demonstrates how to use **hooks** in TinyGent agents. Hooks are callbacks that let you monitor, customize, or intervene in the agent’s lifecycle: LLM calls, tool calls, reasoning steps, final answers, and error handling.

---

## Concept

When you build an agent, you can pass hook functions (like `on_before_llm_call`, `on_after_tool_call`, etc.) that will be executed at specific points during the agent’s operation. This gives you visibility and control without modifying the internal logic.

### Available Hooks

* **on_before_llm_call** – Triggered before an LLM is called.
* **on_after_llm_call** – Triggered after an LLM returns a result.
* **on_before_tool_call** – Triggered before a tool function is executed.
* **on_after_tool_call** – Triggered after a tool returns a result.
* **on_reasoning** – Triggered when the agent produces reasoning steps.
* **on_tool_reasoning** - Triggered when the agent produces reasoning specific to tool usage.
* **on_answer** – Triggered when the agent produces its final answer.
* **on_error** – Triggered when any error occurs.

---

## Why Hooks Matter

* **Debugging**: Inspect inputs, outputs, and tool usage.
* **Logging**: Stream detailed logs to your preferred system.
* **Customization**: Enforce additional checks, metrics, or constraints.
* **Experimentation**: Compare how the agent behaves with different prompts or tool choices.

---

## Example Flow

1. **Agent receives user input** → triggers `on_before_llm_call`.
2. **LLM generates reasoning or a plan** → triggers `on_reasoning`.
3. **Agent decides to call a tool** → triggers `on_before_tool_call` → executes tool → triggers `on_after_tool_call`. -> triggers `on_tool_reasoning` if applicable.
4. **Agent gathers results and formulates answer** → triggers `on_after_llm_call`.
5. **Final answer is produced** → triggers `on_answer`.
6. **If an error occurs** at any stage → triggers `on_error`.

---

## What You’ll See in the Example

* Each hook prints output in a **different color**, using a centralized color utility (`TinyColorPrinter`).
* Logs from the agent lifecycle are mixed with color-coded hook prints, so you can distinguish them clearly.
* A simple `greet` tool shows how tools integrate into this lifecycle.

---

## Takeaway

Hooks provide a **transparent window** into what your agent is doing at each step. They’re especially useful for developers who want to:

* Debug multi-step reasoning
* Trace tool usage
* Capture structured logs
* Or add monitoring without touching the agent’s core logic
