from tinygent.agents.middleware.base import AgentMiddleware
from tinygent.factory.helper import check_modules
from tinygent.runtime.middleware_catalog import GlobalMiddlewareCatalog


def build_middleware(middleware: AgentMiddleware | str) -> AgentMiddleware:
    """Build tiny middleware."""
    check_modules()

    if isinstance(middleware, str):
        return GlobalMiddlewareCatalog.get_active_catalog().get_middleware(middleware)

    return middleware
