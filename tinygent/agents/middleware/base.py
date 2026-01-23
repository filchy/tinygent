from typing import Any
from typing import TypeVar

from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.types.io.llm_io_input import TinyLLMInput

T = TypeVar('T', bound='AgentMiddleware')


class AgentMiddleware:
    def before_llm_call(self, *, run_id: str, llm_input: TinyLLMInput) -> Any:
        pass

    def after_llm_call(
        self, *, run_id: str, llm_input: TinyLLMInput, result: Any
    ) -> Any:
        pass

    def before_tool_call(
        self, *, run_id: str, tool: AbstractTool, args: dict[str, Any]
    ) -> Any:
        pass

    def after_tool_call(
        self, *, run_id: str, tool: AbstractTool, args: dict[str, Any], result: Any
    ) -> Any:
        pass

    def on_plan(self, *, run_id: str, plan: str) -> Any:
        pass

    def on_reasoning(self, *, run_id: str, reasoning: str) -> Any:
        pass

    def on_tool_reasoning(self, *, run_id: str, reasoning: str) -> Any:
        pass

    def on_answer(self, *, run_id: str, answer: str) -> Any:
        pass

    def on_answer_chunk(self, *, run_id: str, chunk: str, idx: str) -> Any:
        pass

    def on_error(self, *, run_id: str, e: Exception) -> Any:
        pass


def register_middleware(name: str):
    def decorator(cls: type[T]) -> type[T]:
        from tinygent.core.runtime.middleware_catalog import GlobalMiddlewareCatalog

        GlobalMiddlewareCatalog().get_active_catalog().register(
            name,
            cls,
        )
        return cls

    return decorator
