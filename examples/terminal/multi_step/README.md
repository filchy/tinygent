# Multi-Step Agent Example

This example demonstrates how to run a **multi-step reasoning agent** with Tinygent in the terminal.

## What is it?

The multi-step agent plans and executes actions iteratively until it reaches a final answer or the maximum step count.
It uses an LLM (OpenAI in this example), configurable planning prompts, and tools like `provide_final_answer` and `get_weather_mock`.

> General terminal usage and CLI options are documented in [`examples/terminal/README.md`](../README.md).

## Config File

The agent is defined in [`config.yaml`](./config.yaml).

```yaml
# examples/terminal/multi_step/config.yaml

type: multistep
max_steps: 10
plan_interval: 2

llm:
  type: openai
  model: gpt-4o-mini
  temperature: 0.4

prompt_template:
  plan:
    init_plan: "Init plan for task {{ task }} using tools {{ tools }}"
    update_plan: "Update plan for task {{ task }} with tools {{ tools }}, history {{ history }}, steps {{ steps }}, remaining steps {{ remaining_steps }}"
  acter:
    system: "System instructions"
    final_answer: "Final answer for task {{ task }} with tools {{ tools }}, history {{ history }}, steps {{ steps }}, tool calls {{ tool_calls }}"
  final:
    final_answer: "Fallback answer for task {{ task }}, history {{ history }}, steps {{ steps }}"

tools:
  - name: provide_final_answer
  - name: get_weather_mock
```

### Key Fields

* **type**: Agent type (`multistep`).
* **max_steps**: Hard cap on reasoning steps before finishing.
* **plan_interval**: Frequency (in steps) for plan updates.
* **llm**: Backend model and generation parameters.
* **prompt_template**: Strings for plan/act/final phases (Jinja-like variables).
* **tools**: Tools the agent can call.

## Run the Example

From the project root, run a single query:

```bash
tiny terminal -c examples/terminal/multi_step/config.yaml -q "What is the weather like in Paris?"
```

Run with debug logging:

```bash
tiny -l debug terminal -c examples/terminal/multi_step/config.yaml -q "What is the weather like in Paris?"
```

Run multiple queries sequentially:

```bash
tiny terminal \
  -c examples/terminal/multi_step/config.yaml \
  -q "What is the weather like in Paris?" \
  -q "What is the weather like in New York?" \
```

## Notes

* Ensure your OpenAI API key is available to the process (e.g., `export OPENAI_API_KEY=...`).
* You can customize prompt templates and swap tools without changing Python code.
* For more agent types, return to the [terminal examples index](../README.md).
