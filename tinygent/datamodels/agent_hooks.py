from abc import ABC
import logging
from typing import Any
from typing import Callable

from tinygent.datamodels.llm_io_input import TinyLLMInput
from tinygent.datamodels.tool import AbstractTool

logger = logging.getLogger(__name__)


def _log_hook(msg: str, level: int = logging.DEBUG) -> None:
    logger.log(level, msg)


class AgentHooks(ABC):
    "Abstract base class for agent hooks to monitor and intervene in the agent's operations."

    def __init__(
        self,
        on_before_llm_call: Callable[[TinyLLMInput], None] | None = None,
        on_after_llm_call: Callable[[TinyLLMInput, Any], None] | None = None,
        on_before_tool_call: (
            Callable[[AbstractTool, dict[str, Any]], None] | None
        ) = None,
        on_after_tool_call: (
            Callable[[AbstractTool, dict[str, Any], Any], None] | None
        ) = None,
        on_plan: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        on_tool_reasoning: Callable[[str], None] | None = None,
        on_answer: Callable[[str], None] | None = None,
        on_answer_chunk: Callable[[str], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.on_before_llm_call: Callable[[TinyLLMInput], None] = on_before_llm_call or (
            lambda llm_input: _log_hook(f'Before LLM call with input: {llm_input}')
        )

        self.on_after_llm_call: Callable[[TinyLLMInput, Any], None] = (
            on_after_llm_call
            or (
                lambda llm_input, result: _log_hook(
                    f'After LLM call. Input: {llm_input}, Result: {result}'
                )
            )
        )

        self.on_before_tool_call: Callable[[AbstractTool, dict[str, Any]], None] = (
            on_before_tool_call
            or (
                lambda tool, args: _log_hook(
                    f'Before tool call: {tool.info.name} with args: {args}'
                )
            )
        )

        self.on_after_tool_call: Callable[[AbstractTool, dict[str, Any], Any], None] = (
            on_after_tool_call
            or (
                lambda tool, args, result: _log_hook(
                    f'After tool call: {tool.info.name} with args: {args}, result: {result}'
                )
            )
        )

        self.on_plan: Callable[[str], None] = on_plan or (
            lambda plan: _log_hook(f'Plan: {plan}')
        )

        self.on_reasoning: Callable[[str], None] = on_reasoning or (
            lambda reasoning: _log_hook(f'Reasoning: {reasoning}')
        )

        self.on_tool_reasoning: Callable[[str], None] = on_tool_reasoning or (
            lambda reasoning: _log_hook(f'Tool Reasoning: {reasoning}')
        )

        self.on_answer: Callable[[str], None] = on_answer or (
            lambda answer: _log_hook(f'Final Answer: {answer}')
        )

        self.on_answer_chunk: Callable[[str], None] = on_answer_chunk or (
            lambda chunk: _log_hook(f'Answer Chunk: {chunk}')
        )

        self.on_error: Callable[[Exception], None] = on_error or (
            lambda e: _log_hook(f'Error occurred: {e}', level=logging.ERROR)
        )
