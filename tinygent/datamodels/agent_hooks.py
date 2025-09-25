from abc import ABC
from typing import Any

from tinygent.datamodels.llm_io import TinyLLMInput
from tinygent.datamodels.tool import AbstractTool


class AgentHooks(ABC):
    """Abstract base class for agent hooks to monitor and intervene in the agent's operations."""

    def on_before_llm_call(self, llm_input: TinyLLMInput) -> None:
        """Called before the LLM is invoked."""
        pass

    def on_after_llm_call(self, llm_input: TinyLLMInput, response: str) -> None:
        """Called after the LLM has returned a response."""
        pass

    def on_before_tool_call(self, tool: AbstractTool, arguments: dict[str, Any]) -> None:
        """Called before a tool is invoked."""
        pass

    def on_after_tool_call(
        self, tool: AbstractTool, arguments: dict[str, Any], result: Any
    ) -> None:
        """Called after a tool has returned a result."""
        pass

    def on_reasoning(self, reasoning: str) -> None:
        """Called when the agent generates reasoning."""
        pass

    def on_answer(self, answer: str) -> None:
        """Called when the agent generates a final answer."""
        pass

    def on_error(self, error: Exception) -> None:
        """Called when an error occurs during LLM or tool invocation."""
        pass
