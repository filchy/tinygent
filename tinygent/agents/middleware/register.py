def _register_middleware() -> None:
    from tinygent.agents.middleware.tool_limiter import ToolCallLimiterMiddleware

    _ = ToolCallLimiterMiddleware


_register_middleware()
