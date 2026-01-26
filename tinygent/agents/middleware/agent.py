from collections.abc import Sequence
import inspect
import logging
from typing import Any

from tinygent.agents.middleware.base import TinyBaseMiddleware
from tinygent.core.datamodels.middleware import AbstractMiddleware
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.types.io.llm_io_input import TinyLLMInput

logger = logging.getLogger(__name__)


class TinyMiddlewareAgent(TinyBaseMiddleware):
    def __init__(self, middleware: Sequence[AbstractMiddleware]) -> None:
        self.middleware = middleware

    @staticmethod
    def _overrides(m: AbstractMiddleware, name: str) -> bool:
        base_attr = getattr(TinyBaseMiddleware, name, None)
        if base_attr is None:
            raise AttributeError(f'{name!r} is not a method of TinyBaseMiddleware')
        return getattr(m.__class__, name) is not base_attr

    async def _dispatch(self, name: str, **kwargs: Any) -> None:
        """Dispatch hook to all middleware. Middleware can mutate the kwargs dict in-place."""
        for m in self.middleware:
            if not self._overrides(m, name):
                continue

            fn = getattr(m, name)
            result = fn(**kwargs)

            if inspect.isawaitable(result):
                await result

    async def before_llm_call(
        self, *, run_id: str, llm_input: TinyLLMInput, kwargs: dict[str, Any]
    ) -> None:
        await self._dispatch(
            'before_llm_call',
            run_id=run_id,
            llm_input=llm_input,
            kwargs=kwargs,
        )

    async def after_llm_call(
        self,
        *,
        run_id: str,
        llm_input: TinyLLMInput,
        result: Any,
        kwargs: dict[str, Any],
    ) -> None:
        await self._dispatch(
            'after_llm_call',
            run_id=run_id,
            llm_input=llm_input,
            result=result,
            kwargs=kwargs,
        )

    async def before_tool_call(
        self,
        *,
        run_id: str,
        tool: AbstractTool,
        args: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> None:
        await self._dispatch(
            'before_tool_call',
            run_id=run_id,
            tool=tool,
            args=args,
            kwargs=kwargs,
        )

    async def after_tool_call(
        self,
        *,
        run_id: str,
        tool: AbstractTool,
        args: dict[str, Any],
        result: Any,
        kwargs: dict[str, Any],
    ) -> None:
        await self._dispatch(
            'after_tool_call',
            run_id=run_id,
            tool=tool,
            args=args,
            result=result,
            kwargs=kwargs,
        )

    async def on_plan(self, *, run_id: str, plan: str, kwargs: dict[str, Any]) -> None:
        await self._dispatch(
            'on_plan',
            run_id=run_id,
            plan=plan,
            kwargs=kwargs,
        )

    async def on_reasoning(
        self, *, run_id: str, reasoning: str, kwargs: dict[str, Any]
    ) -> None:
        await self._dispatch(
            'on_reasoning',
            run_id=run_id,
            reasoning=reasoning,
            kwargs=kwargs,
        )

    async def on_tool_reasoning(
        self, *, run_id: str, reasoning: str, kwargs: dict[str, Any]
    ) -> None:
        await self._dispatch(
            'on_tool_reasoning',
            run_id=run_id,
            reasoning=reasoning,
            kwargs=kwargs,
        )

    async def on_answer(
        self, *, run_id: str, answer: str, kwargs: dict[str, Any]
    ) -> None:
        await self._dispatch(
            'on_answer',
            run_id=run_id,
            answer=answer,
            kwargs=kwargs,
        )

    async def on_answer_chunk(
        self, *, run_id: str, chunk: str, idx: str, kwargs: dict[str, Any]
    ) -> None:
        await self._dispatch(
            'on_answer_chunk',
            run_id=run_id,
            chunk=chunk,
            idx=idx,
            kwargs=kwargs,
        )

    async def on_error(
        self, *, run_id: str, e: Exception, kwargs: dict[str, Any]
    ) -> None:
        await self._dispatch(
            'on_error',
            run_id=run_id,
            e=e,
            kwargs=kwargs,
        )
