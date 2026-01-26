import logging
from typing import Callable

from tinygent.agents.middleware.base import TinyBaseMiddleware

logger = logging.getLogger(__name__)


class MiddlewareCatalog:
    def __init__(self) -> None:
        self._middlewares: dict[str, Callable[[], TinyBaseMiddleware]] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], TinyBaseMiddleware],
    ) -> None:
        logger.debug(
            'Registering middleware %s (%s)',
            name,
            factory.__doc__ or 'Description not defined',
        )

        if name in self._middlewares:
            raise ValueError(f'Middleware {name} already registered.')

        self._middlewares[name] = factory

    def get_middleware(self, name: str) -> TinyBaseMiddleware:
        if name not in self._middlewares:
            raise ValueError(f'Middleware {name} not registered.')

        return self._middlewares[name]()

    def get_middlewares(self) -> list[TinyBaseMiddleware]:
        return [factory() for factory in self._middlewares.values()]


class GlobalMiddlewareCatalog:
    _active_catalog: MiddlewareCatalog = MiddlewareCatalog()

    @staticmethod
    def get_active_catalog() -> MiddlewareCatalog:
        """Get the active global middleware catalog."""
        return GlobalMiddlewareCatalog._active_catalog
