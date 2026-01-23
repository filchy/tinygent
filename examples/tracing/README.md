# Tracing with OpenTelemetry in Tinygent

This example demonstrates how to enable and use OpenTelemetry tracing in Tinygent agents and applications. Tracing allows you to monitor, analyze, and debug the execution flow of your agents by exporting trace data to an OpenTelemetry-compatible backend.

## Prerequisites

You need [Docker](https://www.docker.com/) installed to run the [OpenTelemetry](https://opentelemetry.io/) Collector and [Jaeger](https://www.jaegertracing.io/) backend.

## How to Enable Tracing

To enable tracing, set the following environment variable before running your agent or application:

```bash
export TINY_OTEL_ENABLED=1
```

Optionally, you can specify a custom OpenTelemetry Collector endpoint:

```bash
export TINY_OTEL_COLLECTOR_ENDPOINT="127.0.0.1:4317"
```

By default, the endpoint is set to `127.0.0.1:4317`.

## Usage in Agents

Once tracing is enabled, all agents and core logic in Tinygent that use the tracing decorators or context managers will automatically emit trace data. You do not need to modify your agent code to benefit from tracing—just set the environment variable as shown above.

## Example

The `examples/tracing/main.py` script demonstrates how to create spans, set attributes, and emit nested telemetry using Tinygent utilities.

To run the tracing example:

1. Start the tracing infrastructure:
```bash
cd examples/tracing
docker compose up -d
```

2. Run the telemetry demo script to generate sample spans:
```bash
TINY_OTEL_ENABLED=1 uv run examples/tracing/main.py
```

3. (Optional) Run a full agent with tracing enabled:
```bash
uv sync --extra openai
TINY_OTEL_ENABLED=1 uv run examples/agents/multi-step/main.py
```

4. View traces in the Jaeger UI:
   - Open http://localhost:16686 in your browser
   - Select "tinygent" from the service dropdown
   - Click "Find Traces" to see your agent's execution traces

## How It Works

Tinygent uses OpenTelemetry's Python SDK to instrument code. The tracing is integrated via decorators and context managers, so spans are created for key operations. You can further customize or add tracing in your own code using the utilities in `tinygent.core.telemetry`.

For more details, see the [OpenTelemetry documentation](https://opentelemetry.io/docs/instrumentation/python/).
