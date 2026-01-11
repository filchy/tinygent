from collections.abc import Sequence
from typing import Any

from tinygent.agents.middleware.base import AgentMiddleware
from tinygent.datamodels.tool import AbstractTool
from tinygent.types.io.llm_io_input import TinyLLMInput


class MiddlewareAgent:
    def __init__(self, middleware: Sequence[AgentMiddleware]) -> None:
        self.middleware = middleware

    @staticmethod
    def _overrides(m: AgentMiddleware, name: str) -> bool:
        base_attr = getattr(AgentMiddleware, name, None)
        if base_attr is None:
            raise AttributeError(
                f'{name!r} is not a method of AgentMiddleware'
            )
        return getattr(m.__class__, name) is not base_attr

    def _dispatch(self, name: str, **kwargs: Any) -> None:
        for m in self.middleware:
            if self._overrides(m, name):
                getattr(m, name)(**kwargs)

    def before_llm_call(
        self, *, run_id: str, llm_input: TinyLLMInput
    ) -> None:
        self._dispatch(
            'before_llm_call',
            run_id=run_id,
            llm_input=llm_input,
        )

    def after_llm_call(
        self, *, run_id: str, llm_input: TinyLLMInput, result: Any
    ) -> None:
        self._dispatch(
            'after_llm_call',
            run_id=run_id,
            llm_input=llm_input,
            result=result,
        )

    def before_tool_call(
        self,
        *,
        run_id: str,
        tool: AbstractTool,
        args: dict[str, Any],
    ) -> None:
        self._dispatch(
            'before_tool_call',
            run_id=run_id,
            tool=tool,
            args=args,
        )

    def after_tool_call(
        self,
        *,
        run_id: str,
        tool: AbstractTool,
        args: dict[str, Any],
        result: Any,
    ) -> None:
        self._dispatch(
            'after_tool_call',
            run_id=run_id,
            tool=tool,
            args=args,
            result=result,
        )

    def on_plan(self, *, run_id: str, plan: str) -> None:
        self._dispatch(
            'on_plan',
            run_id=run_id,
            plan=plan,
        )

    def on_reasoning(
        self, *, run_id: str, reasoning: str
    ) -> None:
        self._dispatch(
            'on_reasoning',
            run_id=run_id,
            reasoning=reasoning,
        )

    def on_tool_reasoning(
        self, *, run_id: str, reasoning: str
    ) -> None:
        self._dispatch(
            'on_tool_reasoning',
            run_id=run_id,
            reasoning=reasoning,
        )

    def on_answer(
        self, *, run_id: str, answer: str
    ) -> None:
        self._dispatch(
            'on_answer',
            run_id=run_id,
            answer=answer,
        )

    def on_answer_chunk(
        self, *, run_id: str, chunk: str, idx: str
    ) -> None:
        self._dispatch(
            'on_answer_chunk',
            run_id=run_id,
            chunk=chunk,
            idx=idx,
        )

    def on_error(self, *, run_id: str, e: Exception) -> None:
        self._dispatch(
            'on_error',
            run_id=run_id,
            e=e,
        )
