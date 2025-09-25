from abc import ABC
import logging
from typing import Any, Callable

from tinygent.datamodels.llm_io import TinyLLMInput
from tinygent.datamodels.tool import AbstractTool

logger = logging.getLogger(__name__)


def _log_hook(msg: str, level: int = logging.DEBUG) -> None:
    logger.log(level, msg)


class AgentHooks(ABC):
    ''"Abstract base class for agent hooks to monitor and intervene in the agent's operations."""

    def __init__(self) -> None:
        self.on_before_llm_call: Callable[[TinyLLMInput], None] = (
            lambda llm_input: _log_hook(f"Before LLM call with input: {llm_input}")
        )

        self.on_after_llm_call: Callable[[TinyLLMInput, Any], None] = (
            lambda llm_input, result: _log_hook(f"After LLM call. Input: {llm_input}, Result: {result}")
        )

        self.on_before_tool_call: Callable[[AbstractTool, dict[str, Any]], None] = (
            lambda tool, args: _log_hook(f"Before tool call: {tool.info.name} with args: {args}")
        )

        self.on_after_tool_call: Callable[[AbstractTool, dict[str, Any], Any], None] = (
            lambda tool, args, result: _log_hook(f"After tool call: {tool.info.name} with args: {args}, result: {result}")
        )

        self.on_reasoning: Callable[[str], None] = (
            lambda reasoning: _log_hook(f"Reasoning: {reasoning}")
        )

        self.on_answer: Callable[[str], None] = (
            lambda answer: _log_hook(f"Final Answer: {answer}")
        )

        self.on_error: Callable[[Exception], None] = (
            lambda e: _log_hook(f"Error occurred: {e}", level=logging.ERROR)
        )
