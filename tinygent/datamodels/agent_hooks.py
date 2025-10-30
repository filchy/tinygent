import asyncio
import inspect
import logging
from abc import ABC
from collections.abc import AsyncGenerator
from collections.abc import Generator
from typing import Any
from typing import Coroutine
from typing import Callable
from typing import Optional

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
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            result = fn(*args, **kwargs)

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


class AgentHooks(ABC):
    """
    Abstract base class for agent hooks to monitor and intervene
    in the agent's operations.
    Hooks assigned dynamically are automatically wrapped into
    synchronous safe callables.
    """

    def __init__(
        self,
        on_before_llm_call: Optional[Callable[[TinyLLMInput], Any]] = None,
        on_after_llm_call: Optional[Callable[[TinyLLMInput, Any], Any]] = None,
        on_before_tool_call: Optional[Callable[[AbstractTool, dict[str, Any]], Any]] = None,
        on_after_tool_call: Optional[Callable[[AbstractTool, dict[str, Any], Any], Any]] = None,
        on_plan: Optional[Callable[[str], Any]] = None,
        on_reasoning: Optional[Callable[[str], Any]] = None,
        on_tool_reasoning: Optional[Callable[[str], Any]] = None,
        on_answer: Optional[Callable[[str], Any]] = None,
        on_answer_chunk: Optional[Callable[[str], Any]] = None,
        on_error: Optional[Callable[[Exception], Any]] = None,
    ) -> None:
        # default values
        self._on_before_llm_call = _wrap_hook_sync(
            on_before_llm_call or (lambda llm_input: _log_hook(f'before llm call with input: {llm_input}'))
        )
        self._on_after_llm_call = _wrap_hook_sync(
            on_after_llm_call or (lambda llm_input, result: _log_hook(
                f'after llm call. input: {llm_input}, result: {result}'
            ))
        )
        self._on_before_tool_call = _wrap_hook_sync(
            on_before_tool_call or (lambda tool, args: _log_hook(
                f'before tool call: {tool.info.name} with args: {args}'
            ))
        )
        self._on_after_tool_call = _wrap_hook_sync(
            on_after_tool_call or (lambda tool, args, result: _log_hook(
                f'after tool call: {tool.info.name} with args: {args}, result: {result}'
            ))
        )
        self._on_plan = _wrap_hook_sync(on_plan or (lambda plan: _log_hook(f'plan: {plan}')))
        self._on_reasoning = _wrap_hook_sync(on_reasoning or (lambda r: _log_hook(f'reasoning: {r}')))
        self._on_tool_reasoning = _wrap_hook_sync(
            on_tool_reasoning or (lambda r: _log_hook(f'tool reasoning: {r}'))
        )
        self._on_answer = _wrap_hook_sync(on_answer or (lambda a: _log_hook(f'final answer: {a}')))
        self._on_answer_chunk = _wrap_hook_sync(
            on_answer_chunk or (lambda c: _log_hook(f'answer chunk: {c}'))
        )
        self._on_error = _wrap_hook_sync(
            on_error or (lambda e: _log_hook(f'error occurred: {e}', level=logging.ERROR))
        )

    # region properties with auto-wrap
    @property
    def on_before_llm_call(self): return self._on_before_llm_call
    @on_before_llm_call.setter
    def on_before_llm_call(self, fn): self._on_before_llm_call = _wrap_hook_sync(fn)

    @property
    def on_after_llm_call(self): return self._on_after_llm_call
    @on_after_llm_call.setter
    def on_after_llm_call(self, fn): self._on_after_llm_call = _wrap_hook_sync(fn)

    @property
    def on_before_tool_call(self): return self._on_before_tool_call
    @on_before_tool_call.setter
    def on_before_tool_call(self, fn): self._on_before_tool_call = _wrap_hook_sync(fn)

    @property
    def on_after_tool_call(self): return self._on_after_tool_call
    @on_after_tool_call.setter
    def on_after_tool_call(self, fn): self._on_after_tool_call = _wrap_hook_sync(fn)

    @property
    def on_plan(self): return self._on_plan
    @on_plan.setter
    def on_plan(self, fn): self._on_plan = _wrap_hook_sync(fn)

    @property
    def on_reasoning(self): return self._on_reasoning
    @on_reasoning.setter
    def on_reasoning(self, fn): self._on_reasoning = _wrap_hook_sync(fn)

    @property
    def on_tool_reasoning(self): return self._on_tool_reasoning
    @on_tool_reasoning.setter
    def on_tool_reasoning(self, fn): self._on_tool_reasoning = _wrap_hook_sync(fn)

    @property
    def on_answer(self): return self._on_answer
    @on_answer.setter
    def on_answer(self, fn): self._on_answer = _wrap_hook_sync(fn)

    @property
    def on_answer_chunk(self): return self._on_answer_chunk
    @on_answer_chunk.setter
    def on_answer_chunk(self, fn): self._on_answer_chunk = _wrap_hook_sync(fn)

    @property
    def on_error(self): return self._on_error
    @on_error.setter
    def on_error(self, fn): self._on_error = _wrap_hook_sync(fn)
    # endregion
