# Tool Example: Unified `.run()` for Sync, Async, Generator, Async Generator

This example demonstrates how to use the `@tool` decorator from **tinygent**
to wrap different types of Python functions into a unified interface.

## Features

Each decorated function is converted into a `Tool` instance that:

- Inspects and validates its input and output types
- Automatically generates a Pydantic input/output schema
- Exposes a `.run()` method that works uniformly across all function types

## Behavior of `.run()`

- Awaits async coroutines
- Iterates and collects results from generators and async generators
- Returns plain values for sync functions

This is useful for:

- LLM toolchains
- Plugin systems
- General-purpose dynamic function execution

## Included Functions

- `add(a, b)`: Synchronous addition
- `greet(name)`: Asynchronous greeting
- `count(n)`: Synchronous generator
- `async_count(n)`: Asynchronous generator

## Running the Example

```bash
python main.py
