# Tinygent Terminal Examples

Tinygent includes a **terminal client** that lets you run agents defined in YAML configs directly from the command line. This is especially useful for testing, debugging, and quickly prototyping new agents without writing Python code.

## Usage

From the project root, run:

```bash
tiny terminal -c <path-to-config.yaml> -q "Your query here"
```

### Options

* `-c, --config PATH` – Path to the agent configuration `.yaml` file (required)
* `-q, --query QUERY ...` – One or more queries to run
* `-l, --log-level LEVEL` – Logging verbosity (`debug`, `info`, etc.)

### Example

```bash
tiny -l debug terminal -c examples/terminal/multi_step/config.yaml -q "Jaké je dnes počasí v Přerově?"
```

---

## Available Agents

This folder contains different example agents that can be run in the terminal. Each subfolder has its own `README.md` with configuration details.

* [**Multi-Step Agent**](./multi_stepREADME.md) – Example of a multi-step reasoning agent using OpenAI LLM, planning templates, and weather mock tool.

More agent types will be added here over time.
