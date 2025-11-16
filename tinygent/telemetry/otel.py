from contextlib import contextmanager
import os
from typing import Iterator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Tracer
from opentelemetry.trace import get_current_span
from opentelemetry.util.types import AttributeValue

_tracer_provider: TracerProvider | None = None


def _is_enabled() -> bool:
    return os.getenv('TINY_OTEL_ENABLED', '').lower() in {'1', 'true', 'yes'}


def setup_tiny_otel(service_name: str = 'tinygent') -> Tracer:
    global _tracer_provider

    if not _is_enabled():
        return trace.get_tracer(service_name)

    if _tracer_provider is not None:
        return _tracer_provider.get_tracer(service_name)

    endpoint = os.getenv('TINY_OTEL_COLLECTOR_ENDPOINT', '127.0.0.1:4317')

    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)

    provider = TracerProvider(resource=Resource.create({'service.name': service_name}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    return trace.get_tracer(service_name)


def get_tiny_tracer(service_name: str = 'tinygent') -> Tracer:
    if _tracer_provider is None:
        return setup_tiny_otel(service_name)
    return trace.get_tracer(service_name)


def set_tiny_attribute(key: str, value: AttributeValue) -> None:
    if not _is_enabled():
        return

    span = get_current_span()
    span.set_attribute(key, value)


def set_tiny_attributes(attrs: dict[str, AttributeValue]) -> None:
    if not _is_enabled():
        return

    span = get_current_span()
    for k, v in attrs.items():
        span.set_attribute(k, v)


@contextmanager
def tiny_trace_span(name: str, **attrs: AttributeValue) -> Iterator[trace.Span]:
    tracer = get_tiny_tracer(name)
    with tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            span.set_attribute(k, v)
        yield span
