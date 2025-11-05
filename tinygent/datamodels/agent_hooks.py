from abc import ABC
import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Generator
import inspect
import logging
from typing import Any
from typing import Callable
from typing import Coroutine
from typing import Protocol

from tinygent.datamodels.llm_io_input import TinyLLMInput
from tinygent.datamodels.tool import AbstractTool

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


class HookBeforeLLMCall(Protocol):
    def __call__(self, *, run_id: str, llm_input: TinyLLMInput) -> Any: ...


class HookAfterLLMCall(Protocol):
    def __call__(self, *, run_id: str, llm_input: TinyLLMInput, result: Any) -> Any: ...


class HookBeforeToolCall(Protocol):
    def __call__(
        self, *, run_id: str, tool: AbstractTool, args: dict[str, Any]
    ) -> Any: ...


class HookAfterToolCall(Protocol):
    def __call__(
        self, *, run_id: str, tool: AbstractTool, args: dict[str, Any], result: Any
    ) -> Any: ...


class HookPlan(Protocol):
    def __call__(self, *, run_id: str, plan: str) -> Any: ...


class HookReasoning(Protocol):
    def __call__(self, *, run_id: str, reasoning: str) -> Any: ...


class HookToolReasoning(Protocol):
    def __call__(self, *, run_id: str, reasoning: str) -> Any: ...


class HookAnswer(Protocol):
    def __call__(self, *, run_id: str, answer: str) -> Any: ...


class HookAnswerChunk(Protocol):
    def __call__(self, *, run_id: str, chunk: str, idx: str) -> Any: ...


class HookError(Protocol):
    def __call__(self, *, run_id: str, e: Exception) -> Any: ...


class AgentHooks(ABC):
    """
    Abstract base class for agent hooks to monitor and intervene
    in the agent's operations.
    Hooks assigned dynamically are automatically wrapped into
    synchronous safe callables.
    """

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
    def on_before_llm_call(self, fn: HookBeforeLLMCall) -> None:
        self._on_before_llm_call = _wrap_hook_sync(fn)

    @property
    def on_after_llm_call(self) -> HookAfterLLMCall:
        return self._on_after_llm_call

    @on_after_llm_call.setter
    def on_after_llm_call(self, fn: HookAfterLLMCall) -> None:
        self._on_after_llm_call = _wrap_hook_sync(fn)

    @property
    def on_before_tool_call(self) -> HookBeforeToolCall:
        return self._on_before_tool_call

    @on_before_tool_call.setter
    def on_before_tool_call(self, fn: HookBeforeToolCall) -> None:
        self._on_before_tool_call = _wrap_hook_sync(fn)

    @property
    def on_after_tool_call(self) -> HookAfterToolCall:
        return self._on_after_tool_call

    @on_after_tool_call.setter
    def on_after_tool_call(self, fn: HookAfterToolCall) -> None:
        self._on_after_tool_call = _wrap_hook_sync(fn)

    @property
    def on_plan(self) -> HookPlan:
        return self._on_plan

    @on_plan.setter
    def on_plan(self, fn: HookPlan) -> None:
        self._on_plan = _wrap_hook_sync(fn)

    @property
    def on_reasoning(self) -> HookReasoning:
        return self._on_reasoning

    @on_reasoning.setter
    def on_reasoning(self, fn: HookReasoning) -> None:
        self._on_reasoning = _wrap_hook_sync(fn)

    @property
    def on_tool_reasoning(self) -> HookToolReasoning:
        return self._on_tool_reasoning

    @on_tool_reasoning.setter
    def on_tool_reasoning(self, fn: HookToolReasoning) -> None:
        self._on_tool_reasoning = _wrap_hook_sync(fn)

    @property
    def on_answer(self) -> HookAnswer:
        return self._on_answer

    @on_answer.setter
    def on_answer(self, fn: HookAnswer) -> None:
        self._on_answer = _wrap_hook_sync(fn)

    @property
    def on_answer_chunk(self) -> HookAnswerChunk:
        return self._on_answer_chunk

    @on_answer_chunk.setter
    def on_answer_chunk(self, fn: HookAnswerChunk) -> None:
        self._on_answer_chunk = _wrap_hook_sync(fn)

    @property
    def on_error(self) -> HookError:
        return self._on_error

    @on_error.setter
    def on_error(self, fn: HookError) -> None:
        self._on_error = _wrap_hook_sync(fn)
