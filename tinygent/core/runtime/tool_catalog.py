from __future__ import annotations

import logging
from typing import Any
from typing import Callable

from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.datamodels.tool_info import ToolInfo

logger = logging.getLogger(__name__)


class ToolCatalog:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[Callable[..., Any], bool, int]] = {}
        self._hidden_tools: dict[str, tuple[Callable[..., Any], bool, int]] = {}

    def register(
        self,
        fn: Callable[..., Any],
        *,
        use_cache: bool = False,
        cache_size: int = 128,
        hidden: bool = False,
    ) -> None:
        """Register a new tool function with its cache settings."""
        tool_info = ToolInfo.from_callable(
            fn, use_cache=use_cache, cache_size=cache_size
        )
        name = tool_info.name

        logger.debug('Registering tool %s (hidden=%s)', name, hidden)

        if name in self._tools or name in self._hidden_tools:
            raise ValueError(f'Tool {name} already registered.')

        entry = (fn, use_cache, cache_size)
        if hidden:
            self._hidden_tools[name] = entry
        else:
            self._tools[name] = entry

    def get_tool(self, name: str) -> AbstractTool:
        from tinygent.tools.tool import Tool

        """Return a fresh Tool instance by name."""
        if name in self._tools:
            fn, use_cache, cache_size = self._tools[name]
            return Tool(fn, use_cache=use_cache, cache_size=cache_size)
        if name in self._hidden_tools:
            fn, use_cache, cache_size = self._hidden_tools[name]
            return Tool(fn, use_cache=use_cache, cache_size=cache_size)
        raise ValueError(f'Tool {name} not registered.')

    def get_tools(self, include_hidden: bool = False) -> dict[str, AbstractTool]:
        from tinygent.tools.tool import Tool

        """Return fresh Tool instances for all registered tools."""
        entries = (
            {**self._tools, **self._hidden_tools} if include_hidden else self._tools
        )
        return {
            name: Tool(fn, use_cache=use_cache, cache_size=cache_size)
            for name, (fn, use_cache, cache_size) in entries.items()
        }


class GlobalToolCatalog:
    _active_catalog: ToolCatalog = ToolCatalog()

    @staticmethod
    def get_active_catalog() -> ToolCatalog:
        """Get the active global tool catalog."""
        return GlobalToolCatalog._active_catalog
