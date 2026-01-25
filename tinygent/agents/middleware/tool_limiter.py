import logging
from typing import Any

from tinygent.agents.middleware.base import AgentMiddleware
from tinygent.agents.middleware.base import register_middleware
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.types.io.llm_io_input import TinyLLMInput

logger = logging.getLogger(__name__)


class ToolCallBlockedException(Exception):
    """Exception raised when a tool call is blocked by the limiter."""

    pass


@register_middleware('tool_limiter')
class ToolCallLimiterMiddleware(AgentMiddleware):
    """Middleware that limits tool calls per run.

    Can operate in two modes:
    - Global limiter: Limits all tool calls (tool_name=None)
    - Single tool limiter: Limits specific tool by name (tool_name="tool_name")

    When the limit is reached, the behavior depends on hard_block:
    - hard_block=True: Blocks tool execution and returns error result
    - hard_block=False: Allows execution but adds system message asking LLM to stop

    Args:
        tool_name: Specific tool to limit (None = limit all tools globally)
        max_tool_calls: Maximum number of tool calls allowed per run
        hard_block: Whether to hard block (True) or soft limit (False) tool calls
    """

    def __init__(
        self,
        *,
        tool_name: str | None = None,
        max_tool_calls: int = 10,
        hard_block: bool = True,
    ) -> None:
        self.tool_name = tool_name
        self.max_tool_calls = max_tool_calls
        self.hard_block = hard_block
        self.tool_call_counts: dict[str, int] = {}
        self.limit_reached: dict[str, bool] = {}

    def _check_target_tool(self, tool: AbstractTool) -> bool:
        if not self.tool_name:
            return True
        return self.tool_name == tool.info.name

    def before_tool_call(
        self, *, run_id: str, tool: AbstractTool, args: dict[str, Any]
    ) -> None:
        if not self._check_target_tool(tool):
            return

        current_count = self.tool_call_counts.get(run_id, 0)

        if current_count >= self.max_tool_calls:
            if self.hard_block:
                logger.error(
                    f'Tool call blocked: {tool.info.name} '
                    f'(limit: {self.max_tool_calls}, current: {current_count})'
                )
                raise ToolCallBlockedException(
                    f'Tool call limit reached ({current_count}/{self.max_tool_calls}). '
                    f'Tool "{tool.info.name}" was blocked.'
                )
            else:
                logger.warning(
                    f'Tool call over limit: {tool.info.name} '
                    f'(limit: {self.max_tool_calls}, current: {current_count}) '
                    f'- soft limit mode, allowing execution'
                )

        self.tool_call_counts[run_id] = current_count + 1

        if current_count + 1 >= self.max_tool_calls:
            self.limit_reached[run_id] = True
            logger.warning(
                f'Tool call {self.tool_call_counts[run_id]}/{self.max_tool_calls}: '
                f'{tool.info.name} (LIMIT REACHED - no more tools after this)'
            )
        else:
            logger.debug(
                f'Tool call {self.tool_call_counts[run_id]}/{self.max_tool_calls}: '
                f'{tool.info.name}'
            )

    def before_llm_call(self, *, run_id: str, llm_input: TinyLLMInput) -> None:
        if not self.hard_block:
            current_count = self.tool_call_counts.get(run_id, 0)

            if current_count >= self.max_tool_calls:
                from tinygent.core.datamodels.messages import TinySystemMessage

                limit_message = TinySystemMessage(
                    content=(
                        f'IMPORTANT: You have reached the tool call limit '
                        f'({current_count}/{self.max_tool_calls}). '
                        f'You must now provide a final answer based on the '
                        f'information you have already gathered. Do not attempt to '
                        f'use any more tools. Provide your best answer with the '
                        f'available information.'
                    )
                )
                llm_input.add_at_end(limit_message)

    def on_answer(self, *, run_id: str, answer: str) -> None:
        if run_id in self.tool_call_counts:
            total = self.tool_call_counts[run_id]
            logger.debug(f'Run completed with {total} tool calls')
            del self.tool_call_counts[run_id]

        if run_id in self.limit_reached:
            del self.limit_reached[run_id]

    def on_error(self, *, run_id: str, e: Exception) -> None:
        if run_id in self.tool_call_counts:
            del self.tool_call_counts[run_id]

        if run_id in self.limit_reached:
            del self.limit_reached[run_id]

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about tool call usage."""
        return {
            'tool_name': self.tool_name,
            'max_tool_calls': self.max_tool_calls,
            'hard_block': self.hard_block,
            'active_runs': len(self.tool_call_counts),
            'current_counts': self.tool_call_counts.copy(),
            'runs_at_limit': sum(1 for v in self.limit_reached.values() if v),
        }
