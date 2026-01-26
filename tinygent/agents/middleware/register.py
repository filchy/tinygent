from tinygent.agents.middleware.llm_tool_selector import TinyLLMToolSelectorMiddleware
from tinygent.agents.middleware.llm_tool_selector import (
    TinyLLMToolSelectorMiddlewareConfig,
)
from tinygent.agents.middleware.tool_limiter import TinyToolCallLimiterMiddleware
from tinygent.agents.middleware.tool_limiter import TinyToolCallLimiterMiddlewareConfig
from tinygent.core.runtime.global_registry import GlobalRegistry


def _register_middleware() -> None:
    registry = GlobalRegistry().get_registry()

    registry.register_middleware(
        'tool_limiter',
        TinyToolCallLimiterMiddlewareConfig,
        TinyToolCallLimiterMiddleware,
    )
    registry.register_middleware(
        'llm_tool_selector',
        TinyLLMToolSelectorMiddlewareConfig,
        TinyLLMToolSelectorMiddleware,
    )


_register_middleware()
