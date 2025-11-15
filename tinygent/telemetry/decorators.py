from functools import wraps
from inspect import isasyncgenfunction
from inspect import iscoroutinefunction
from inspect import isgeneratorfunction
from typing import Any

from .otel import _is_enabled
from .otel import get_tiny_tracer


def tiny_trace(name: str | None = None):
    def _wrapper(func):
        span_name = name or func.__name__

        if isasyncgenfunction(func):

            @wraps(func)
            async def _inner_async_gen(*args: Any, **kwargs: Any):
                if not _is_enabled():
                    async for item in func(*args, **kwargs):
                        yield item
                    return

                tracer = get_tiny_tracer()
                with tracer.start_as_current_span(span_name):
                    async for item in func(*args, **kwargs):
                        yield item

            return _inner_async_gen

        if isgeneratorfunction(func):

            @wraps(func)
            def _inner_gen(*args: Any, **kwargs: Any):
                if not _is_enabled():
                    yield from func(*args, **kwargs)
                    return

                tracer = get_tiny_tracer()
                with tracer.start_as_current_span(span_name):
                    yield from func(*args, **kwargs)

            return _inner_gen

        if iscoroutinefunction(func):

            @wraps(func)
            async def _inner_async(*args: Any, **kwargs: Any):
                if not _is_enabled():
                    return await func(*args, **kwargs)

                tracer = get_tiny_tracer()
                with tracer.start_as_current_span(span_name):
                    return await func(*args, **kwargs)

            return _inner_async

        @wraps(func)
        def _inner_sync(*args: Any, **kwargs: Any):
            if not _is_enabled():
                return func(*args, **kwargs)

            tracer = get_tiny_tracer()
            with tracer.start_as_current_span(span_name):
                return func(*args, **kwargs)

        return _inner_sync

    return _wrapper
