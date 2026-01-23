from tinygent.core.runtime.global_registry import GlobalRegistry
from tinygent.tools.jit_tool import JITInstructionTool
from tinygent.tools.jit_tool import JITInstructionToolConfig
from tinygent.tools.reasoning_tool import ReasoningTool
from tinygent.tools.reasoning_tool import ReasoningToolConfig
from tinygent.tools.tool import Tool
from tinygent.tools.tool import ToolConfig


def _register_tools() -> None:
    registry = GlobalRegistry().get_registry()

    registry.register_tool('simple', ToolConfig, Tool)
    registry.register_tool('reasoning', ReasoningToolConfig, ReasoningTool)
    registry.register_tool('jit', JITInstructionToolConfig, JITInstructionTool)


_register_tools()
