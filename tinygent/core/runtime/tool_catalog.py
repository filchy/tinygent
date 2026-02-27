from __future__ import annotations

import logging
from typing import Callable

from tinygent.core.datamodels.tool import AbstractTool

logger = logging.getLogger(__name__)


class ToolCatalog:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[[], AbstractTool]] = {}
        self._hidden_tools: dict[str, Callable[[], AbstractTool]] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], AbstractTool],
        *,
        hidden: bool = False,
    ) -> None:
        """Register a tool factory by name."""
        logger.debug('Registering tool %s (hidden=%s)', name, hidden)

        if name in self._tools or name in self._hidden_tools:
            raise ValueError(f'Tool {name} already registered.')

        if hidden:
            self._hidden_tools[name] = factory
        else:
            self._tools[name] = factory

    def get_tool(self, name: str) -> AbstractTool:
        """Return a fresh Tool instance by name."""
        if name in self._tools:
            return self._tools[name]()
        if name in self._hidden_tools:
            return self._hidden_tools[name]()
        raise ValueError(f'Tool {name} not registered.')

    def get_tools(self, include_hidden: bool = False) -> dict[str, AbstractTool]:
        """Return fresh Tool instances for all registered tools."""
        entries = (
            {**self._tools, **self._hidden_tools} if include_hidden else self._tools
        )
        return {name: factory() for name, factory in entries.items()}


class GlobalToolCatalog:
    _active_catalog: ToolCatalog = ToolCatalog()

    @staticmethod
    def get_active_catalog() -> ToolCatalog:
        """Get the active global tool catalog."""
        return GlobalToolCatalog._active_catalog
