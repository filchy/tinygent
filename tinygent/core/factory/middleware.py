import logging
from typing import overload

from tinygent.core.datamodels.middleware import AbstractMiddleware
from tinygent.core.datamodels.middleware import AbstractMiddlewareConfig
from tinygent.core.factory.helper import check_modules
from tinygent.core.factory.helper import parse_config
from tinygent.core.runtime.global_registry import GlobalRegistry
from tinygent.core.runtime.middleware_catalog import GlobalMiddlewareCatalog

logger = logging.getLogger(__name__)


@overload
def build_middleware(
    middleware: AbstractMiddleware,
) -> AbstractMiddleware: ...


@overload
def build_middleware(
    middleware: AbstractMiddlewareConfig,
) -> AbstractMiddleware: ...


@overload
def build_middleware(
    middleware: dict,
) -> AbstractMiddleware: ...


@overload
def build_middleware(
    middleware: str,
    **kwargs,
) -> AbstractMiddleware: ...


def build_middleware(
    middleware: dict | AbstractMiddleware | AbstractMiddlewareConfig | str, **kwargs
) -> AbstractMiddleware:
    """Build tiny middleware."""
    check_modules()

    if isinstance(middleware, AbstractMiddleware):
        return middleware

    if isinstance(middleware, str):
        name = middleware
        try:
            middleware_dict = {'type': name, **kwargs}
            middleware_config = parse_config(
                middleware_dict,
                lambda: GlobalRegistry.get_registry().get_middlewares(),
            )
            return middleware_config.build()
        except ValueError:
            logger.warning(
                "Middleware '%s' not found in global type registry, checking middleware catalog.",
                middleware,
            )
            return GlobalMiddlewareCatalog.get_active_catalog().get_middleware(name)

    if isinstance(middleware, AbstractMiddlewareConfig):
        middleware = middleware.model_dump()

    middleware_config = parse_config(
        middleware, lambda: GlobalRegistry.get_registry().get_middlewares()
    )
    return middleware_config.build()
