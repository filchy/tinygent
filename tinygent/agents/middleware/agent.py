from collections.abc import Sequence
import inspect
import logging
from typing import Any

from tinygent.agents.middleware.base import AgentMiddleware
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.types.io.llm_io_input import TinyLLMInput

logger = logging.getLogger(__name__)


class MiddlewareAgent:
    def __init__(self, middleware: Sequence[AgentMiddleware]) -> None:
        self.middleware = middleware

    @staticmethod
    def _overrides(m: AgentMiddleware, name: str) -> bool:
        base_attr = getattr(AgentMiddleware, name, None)
        if base_attr is None:
            raise AttributeError(f'{name!r} is not a method of AgentMiddleware')
        return getattr(m.__class__, name) is not base_attr

    async def _dispatch(self, name: str, **kwargs: Any) -> None:
        for m in self.middleware:
            if not self._overrides(m, name):
                continue

            fn = getattr(m, name)
            result = fn(**kwargs)

            if inspect.isawaitable(result):
                await result

    async def before_llm_call(self, *, run_id: str, llm_input: TinyLLMInput) -> None:
        await self._dispatch(
            'before_llm_call',
            run_id=run_id,
            llm_input=llm_input,
        )

    async def after_llm_call(
        self, *, run_id: str, llm_input: TinyLLMInput, result: Any
    ) -> None:
        await self._dispatch(
            'after_llm_call',
            run_id=run_id,
            llm_input=llm_input,
            result=result,
        )

    async def before_tool_call(
        self,
        *,
        run_id: str,
        tool: AbstractTool,
        args: dict[str, Any],
    ) -> None:
        await self._dispatch(
            'before_tool_call',
            run_id=run_id,
            tool=tool,
            args=args,
        )

    async def after_tool_call(
        self,
        *,
        run_id: str,
        tool: AbstractTool,
        args: dict[str, Any],
        result: Any,
    ) -> None:
        await self._dispatch(
            'after_tool_call',
            run_id=run_id,
            tool=tool,
            args=args,
            result=result,
        )

    async def on_plan(self, *, run_id: str, plan: str) -> None:
        await self._dispatch(
            'on_plan',
            run_id=run_id,
            plan=plan,
        )

    async def on_reasoning(self, *, run_id: str, reasoning: str) -> None:
        await self._dispatch(
            'on_reasoning',
            run_id=run_id,
            reasoning=reasoning,
        )

    async def on_tool_reasoning(self, *, run_id: str, reasoning: str) -> None:
        await self._dispatch(
            'on_tool_reasoning',
            run_id=run_id,
            reasoning=reasoning,
        )

    async def on_answer(self, *, run_id: str, answer: str) -> None:
        await self._dispatch(
            'on_answer',
            run_id=run_id,
            answer=answer,
        )

    async def on_answer_chunk(self, *, run_id: str, chunk: str, idx: str) -> None:
        await self._dispatch(
            'on_answer_chunk',
            run_id=run_id,
            chunk=chunk,
            idx=idx,
        )

    async def on_error(self, *, run_id: str, e: Exception) -> None:
        await self._dispatch(
            'on_error',
            run_id=run_id,
            e=e,
        )
