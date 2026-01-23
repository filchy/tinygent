from typing import Any
from typing import Protocol

from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.types.io.llm_io_input import TinyLLMInput


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
