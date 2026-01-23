from typing import overload

from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.datamodels.tool import AbstractToolConfig
from tinygent.core.factory.helper import check_modules
from tinygent.core.factory.helper import parse_config
from tinygent.core.runtime.global_registry import GlobalRegistry


@overload
def build_tool(tool: dict | AbstractToolConfig) -> AbstractTool: ...


@overload
def build_tool(tool: str) -> AbstractTool: ...


@overload
def build_tool(tool: str, *, tool_type: str) -> AbstractTool: ...


def build_tool(
    tool: dict | AbstractToolConfig | str, *, tool_type: str | None = None, **tool_kargs
) -> AbstractTool:
    """Build tiny tool."""
    check_modules()

    if isinstance(tool, str):
        tool_type = tool_type if tool_type else 'simple'
        tool = {'name': tool, 'type': tool_type, **tool_kargs}

    if isinstance(tool, AbstractToolConfig):
        tool = tool.model_dump()

    tool_config = parse_config(tool, lambda: GlobalRegistry.get_registry().get_tools())
    return tool_config.build()
