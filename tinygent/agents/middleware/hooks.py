import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Generator
import inspect
import logging
from typing import Any
from typing import Callable
from typing import Coroutine

from tinygent.datamodels.agent_hooks import HookAfterLLMCall
from tinygent.datamodels.agent_hooks import HookAfterToolCall
from tinygent.datamodels.agent_hooks import HookAnswer
from tinygent.datamodels.agent_hooks import HookAnswerChunk
from tinygent.datamodels.agent_hooks import HookBeforeLLMCall
from tinygent.datamodels.agent_hooks import HookBeforeToolCall
from tinygent.datamodels.agent_hooks import HookError
from tinygent.datamodels.agent_hooks import HookPlan
from tinygent.datamodels.agent_hooks import HookReasoning
from tinygent.datamodels.agent_hooks import HookToolReasoning

logger = logging.getLogger(__name__)


def _log_hook(msg: str, level: int = logging.DEBUG) -> None:
    logger.log(level, msg)


def _run_async_nowait(coro: Coroutine[Any, Any, Any]) -> None:
    """Safely schedule an async coroutine, even if called from a running loop."""

    async def _guard():
        try:
            await coro
        except Exception as e:
            logger.exception('Error in async hook: %s', e)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_guard())
    else:
        loop.create_task(_guard())


def _wrap_hook_sync(fn: Callable[..., Any]) -> Callable[..., None]:
    def wrapper(*_args, **kwargs: Any) -> None:
        try:
            result = fn(**kwargs)

            if inspect.isawaitable(result):

                async def _run():
                    await result

                _run_async_nowait(_run())

            elif isinstance(result, Generator):
                for _ in result:
                    pass

            elif isinstance(result, AsyncGenerator):

                async def exhaust():
                    async for _ in result:
                        pass

                _run_async_nowait(exhaust())

        except Exception as e:
            logger.exception('Error in hook %s: %s', fn, e)

    return wrapper


class AgentHooks:
    """Agent hook container with default logging behavior."""

    def __init__(
        self,
        on_before_llm_call: HookBeforeLLMCall | None = None,
        on_after_llm_call: HookAfterLLMCall | None = None,
        on_before_tool_call: HookBeforeToolCall | None = None,
        on_after_tool_call: HookAfterToolCall | None = None,
        on_plan: HookPlan | None = None,
        on_reasoning: HookReasoning | None = None,
        on_tool_reasoning: HookToolReasoning | None = None,
        on_answer: HookAnswer | None = None,
        on_answer_chunk: HookAnswerChunk | None = None,
        on_error: HookError | None = None,
    ) -> None:
        self._on_before_llm_call = _wrap_hook_sync(
            on_before_llm_call
            or (
                lambda *, run_id, llm_input: _log_hook(
                    f'[{run_id}] before LLM call: {llm_input}'
                )
            )
        )
        self._on_after_llm_call = _wrap_hook_sync(
            on_after_llm_call
            or (
                lambda *, run_id, llm_input, result: _log_hook(
                    f'[{run_id}] after LLM call → {result}'
                )
            )
        )
        self._on_before_tool_call = _wrap_hook_sync(
            on_before_tool_call
            or (
                lambda *, run_id, tool, args: _log_hook(
                    f'[{run_id}] before tool {tool.info.name}: {args}'
                )
            )
        )
        self._on_after_tool_call = _wrap_hook_sync(
            on_after_tool_call
            or (
                lambda *, run_id, tool, args, result: _log_hook(
                    f'[{run_id}] after tool {tool.info.name}: {result}'
                )
            )
        )
        self._on_plan = _wrap_hook_sync(
            on_plan or (lambda *, run_id, plan: _log_hook(f'[{run_id}] plan: {plan}'))
        )
        self._on_reasoning = _wrap_hook_sync(
            on_reasoning
            or (
                lambda *, run_id, reasoning: _log_hook(
                    f'[{run_id}] reasoning: {reasoning}'
                )
            )
        )
        self._on_tool_reasoning = _wrap_hook_sync(
            on_tool_reasoning
            or (
                lambda *, run_id, reasoning: _log_hook(
                    f'[{run_id}] tool reasoning: {reasoning}'
                )
            )
        )
        self._on_answer = _wrap_hook_sync(
            on_answer
            or (
                lambda *, run_id, answer: _log_hook(f'[{run_id}] final answer: {answer}')
            )
        )
        self._on_answer_chunk = _wrap_hook_sync(
            on_answer_chunk
            or (
                lambda *, run_id, chunk, idx: _log_hook(
                    f'[{run_id}] answer chunk [{idx}]: {chunk}'
                )
            )
        )
        self._on_error = _wrap_hook_sync(
            on_error
            or (
                lambda *, run_id, e: _log_hook(
                    f'[{run_id}] error occurred: {e}', level=logging.ERROR
                )
            )
        )

    @property
    def on_before_llm_call(self) -> HookBeforeLLMCall:
        return self._on_before_llm_call

    @on_before_llm_call.setter
    def on_before_llm_call(self, fn: HookBeforeLLMCall | None) -> None:
        if fn is None:
            self._on_before_llm_call = lambda *args, **kwargs: None
            return
        self._on_before_llm_call = _wrap_hook_sync(fn)

    @property
    def on_after_llm_call(self) -> HookAfterLLMCall:
        return self._on_after_llm_call

    @on_after_llm_call.setter
    def on_after_llm_call(self, fn: HookAfterLLMCall | None) -> None:
        if fn is None:
            self._on_after_llm_call = lambda *args, **kwargs: None
            return
        self._on_after_llm_call = _wrap_hook_sync(fn)

    @property
    def on_before_tool_call(self) -> HookBeforeToolCall:
        return self._on_before_tool_call

    @on_before_tool_call.setter
    def on_before_tool_call(self, fn: HookBeforeToolCall | None) -> None:
        if fn is None:
            self._on_before_tool_call = lambda *args, **kwargs: None
            return
        self._on_before_tool_call = _wrap_hook_sync(fn)

    @property
    def on_after_tool_call(self) -> HookAfterToolCall:
        return self._on_after_tool_call

    @on_after_tool_call.setter
    def on_after_tool_call(self, fn: HookAfterToolCall | None) -> None:
        if fn is None:
            self._on_after_tool_call = lambda *args, **kwargs: None
            return
        self._on_after_tool_call = _wrap_hook_sync(fn)

    @property
    def on_plan(self) -> HookPlan:
        return self._on_plan

    @on_plan.setter
    def on_plan(self, fn: HookPlan | None) -> None:
        if fn is None:
            self._on_plan = lambda *args, **kwargs: None
            return
        self._on_plan = _wrap_hook_sync(fn)

    @property
    def on_reasoning(self) -> HookReasoning:
        return self._on_reasoning

    @on_reasoning.setter
    def on_reasoning(self, fn: HookReasoning | None) -> None:
        if fn is None:
            self._on_reasoning = lambda *args, **kwargs: None
            return
        self._on_reasoning = _wrap_hook_sync(fn)

    @property
    def on_tool_reasoning(self) -> HookToolReasoning:
        return self._on_tool_reasoning

    @on_tool_reasoning.setter
    def on_tool_reasoning(self, fn: HookToolReasoning | None) -> None:
        if fn is None:
            self._on_tool_reasoning = lambda *args, **kwargs: None
            return
        self._on_tool_reasoning = _wrap_hook_sync(fn)

    @property
    def on_answer(self) -> HookAnswer:
        return self._on_answer

    @on_answer.setter
    def on_answer(self, fn: HookAnswer | None) -> None:
        if fn is None:
            self._on_answer = lambda *args, **kwargs: None
            return
        self._on_answer = _wrap_hook_sync(fn)

    @property
    def on_answer_chunk(self) -> HookAnswerChunk:
        return self._on_answer_chunk

    @on_answer_chunk.setter
    def on_answer_chunk(self, fn: HookAnswerChunk | None) -> None:
        if fn is None:
            self._on_answer_chunk = lambda *args, **kwargs: None
            return
        self._on_answer_chunk = _wrap_hook_sync(fn)

    @property
    def on_error(self) -> HookError:
        return self._on_error

    @on_error.setter
    def on_error(self, fn: HookError | None) -> None:
        if fn is None:
            self._on_error = lambda *args, **kwargs: None
            return
        self._on_error = _wrap_hook_sync(fn)
