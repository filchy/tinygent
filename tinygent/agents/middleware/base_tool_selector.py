from dataclasses import dataclass
import logging
from typing import Any
from typing import Generic
from typing import TypeVar
from typing import cast

from pydantic import Field

from tinygent.agents.middleware.base import TinyBaseMiddleware
from tinygent.agents.middleware.base import TinyBaseMiddlewareConfig
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.datamodels.tool import AbstractToolConfig
from tinygent.core.factory import build_tool

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='TinyBaseToolSelectorMiddleware')


@dataclass
class ToolSelectorCandidates:
    selected_tools: set[AbstractTool]
    mapped_tools: dict[str, AbstractTool]
    tools: list[AbstractTool]
    remaining_tools: list[AbstractTool]
    remaining_space: int | None


class TinyBaseToolSelectorMiddlewareConfig(
    TinyBaseMiddlewareConfig[T],
    Generic[T],
):
    """Configuration for BaseToolSelector"""

    max_tools: int | None = Field(default=None)

    always_include: list[str | AbstractTool | AbstractToolConfig] | None = Field(
        default=None
    )

    def build_base_kwargs(self) -> dict:
        always_include: list[AbstractTool] | None = None

        if self.always_include:
            always_include = [
                t if isinstance(t, AbstractTool) else build_tool(t)
                for t in self.always_include
            ]

        return {
            'max_tools': self.max_tools,
            'always_include': always_include,
        }


class TinyBaseToolSelectorMiddleware(TinyBaseMiddleware):
    def __init__(
        self,
        max_tools: int | None = None,
        always_include: list[AbstractTool] | None = None,
    ) -> None:
        self.max_tools = max_tools
        self.always_include = always_include

        if always_include and max_tools and len(always_include) > max_tools:
            logger.warning(
                'always_include contains %d items which exceeds max_tools=%d; '
                'increasing max_tools to %d to fit always_include.',
                len(always_include),
                max_tools,
                len(always_include),
            )
            self.max_tools = len(always_include)

    def _prepare_candidates(
        self, kwargs: dict[str, Any]
    ) -> ToolSelectorCandidates | None:
        """Build the initial candidate set from always_include and check max_tools.

        Returns None in two cases:
        - No tools are present in kwargs (nothing to select from).
        - max_tools is set and already exhausted by always_include tools alone
          (kwargs['tools'] is updated in place before returning None).

        Otherwise returns a ToolSelectorCandidates with:
        - selected_tools: tools pinned via always_include
        - mapped_tools: name → tool mapping for all available tools
        - tools: full list of available tools
        - remaining_space: slots left after always_include (None = unlimited)
        """
        if not kwargs.get('tools'):
            return None

        selected_tools: set[AbstractTool] = set()
        tools = cast(list[AbstractTool], kwargs.get('tools', []))
        mapped_tools: dict[str, AbstractTool] = {t.info.name: t for t in tools}

        if self.always_include:
            for tool in self.always_include:
                if t := mapped_tools.get(tool.info.name):
                    selected_tools.add(t)

        remaining_space: int | None = None
        if self.max_tools:
            remaining_space = self.max_tools - len(selected_tools)
            if remaining_space <= 0:
                kwargs['tools'] = selected_tools
                return None

        return ToolSelectorCandidates(
            selected_tools=selected_tools,
            mapped_tools=mapped_tools,
            tools=tools,
            remaining_tools=[t for t in tools if t not in selected_tools],
            remaining_space=remaining_space,
        )
