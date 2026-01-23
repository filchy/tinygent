from __future__ import annotations

import os
import time
from typing import Any

from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.otel import set_tiny_attribute
from tinygent.core.telemetry.otel import set_tiny_attributes
from tinygent.core.telemetry.otel import setup_tiny_otel
from tinygent.core.telemetry.otel import tiny_trace_span


@tiny_trace('load-config')
def load_config() -> dict[str, Any]:
    """Simulate loading configuration data for an agent."""
    config = {'agent_name': 'demo-agent', 'step_count': 3}
    set_tiny_attributes(
        {
            'config.agent_name': config['agent_name'],
            'config.step_count': config['step_count'],
        }
    )
    time.sleep(0.05)
    return config


@tiny_trace('process-step')
def process_step(step_index: int) -> None:
    """Simulate a unit of work and annotate the active span."""
    set_tiny_attribute('step.index', step_index)
    time.sleep(0.05)


def main() -> None:
    if os.getenv('TINY_OTEL_ENABLED') is None:
        print('Tracing is disabled. Set TINY_OTEL_ENABLED=1 to export spans.')

    setup_tiny_otel('tinygent-tracing-example')

    with tiny_trace_span('demo-run', example='telemetry'):
        config = load_config()
        for idx in range(config['step_count']):
            process_step(idx)

        with tiny_trace_span('finalize', status='ok'):
            time.sleep(0.05)

    print(
        'Demo completed. If tracing is enabled, open http://localhost:16686 to inspect spans.'
    )


if __name__ == '__main__':
    main()
