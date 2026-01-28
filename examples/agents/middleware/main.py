import time
from typing import Any

from pydantic import Field

from tinygent.agents import TinyMultiStepAgent
from tinygent.agents.middleware import TinyBaseMiddleware
from tinygent.agents.middleware import register_middleware
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.factory import build_llm
from tinygent.core.types import TinyLLMInput
from tinygent.core.types import TinyModel
from tinygent.memory import BufferChatMemory
from tinygent.prompts.multistep import ActionPromptTemplate
from tinygent.prompts.multistep import FallbackAnswerPromptTemplate
from tinygent.prompts.multistep import MultiStepPromptTemplate
from tinygent.prompts.multistep import PlanPromptTemplate
from tinygent.tools import reasoning_tool
from tinygent.utils import TinyColorPrinter


class GreetInput(TinyModel):
    name: str = Field(..., description='The name of the person to greet.')


class CalculateInput(TinyModel):
    a: int = Field(..., description='First number')
    b: int = Field(..., description='Second number')


@reasoning_tool
def greet(data: GreetInput) -> str:
    """Return a simple greeting."""
    return f'Hello, {data.name}!'


@reasoning_tool
def add_numbers(data: CalculateInput) -> str:
    """Add two numbers together."""
    result = data.a + data.b
    return f'The sum of {data.a} and {data.b} is {result}'


@register_middleware('plan_tracker')
class PlanTrackingMiddleware(TinyBaseMiddleware):
    """Middleware that tracks planning steps."""

    def __init__(self) -> None:
        self.plans: list[str] = []

    async def on_plan(self, *, run_id: str, plan: str, kwargs: dict[str, Any]) -> None:
        self.plans.append(plan)
        print(
            TinyColorPrinter.custom(
                'PLAN STEP',
                f'[Run: {run_id[:8]}...] {plan}',
                color='YELLOW',
            )
        )

    def get_all_plans(self) -> list[str]:
        """Return all collected plans."""
        return self.plans


@register_middleware('answer_logger')
class AnswerLoggingMiddleware(TinyBaseMiddleware):
    """Middleware that logs the final answer with formatting."""

    def __init__(self) -> None:
        self.answers: list[str] = []

    async def on_answer(
        self, *, run_id: str, answer: str, kwargs: dict[str, Any]
    ) -> None:
        self.answers.append(answer)
        print(
            TinyColorPrinter.custom(
                'FINAL ANSWER',
                f'[Run: {run_id[:8]}...]\n{answer}',
                color='GREEN',
            )
        )

    async def on_answer_chunk(
        self, *, run_id: str, chunk: str, idx: str, kwargs: dict[str, Any]
    ) -> None:
        # For streaming responses
        print(
            TinyColorPrinter.custom(
                'ANSWER CHUNK',
                f'[{idx}] {chunk}',
                color='CYAN',
            )
        )

    def get_all_answers(self) -> list[str]:
        """Return all collected answers."""
        return self.answers


@register_middleware('llm_timing')
class LLMCallTimingMiddleware(TinyBaseMiddleware):
    """Middleware that tracks LLM call timing and statistics."""

    def __init__(self) -> None:
        self.call_start_times: dict[str, float] = {}
        self.call_durations: list[float] = []
        self.total_calls = 0

    async def before_llm_call(
        self, *, run_id: str, llm_input: TinyLLMInput, kwargs: dict[str, Any]
    ) -> None:
        self.call_start_times[run_id] = time.time()
        self.total_calls += 1

        # Count messages in input
        message_count = len(llm_input.messages) if llm_input.messages else 0

        print(
            TinyColorPrinter.custom(
                'LLM CALL START',
                f'[Run: {run_id[:8]}...] Call #{self.total_calls} | Messages: {message_count}',
                color='BLUE',
            )
        )

    async def after_llm_call(
        self,
        *,
        run_id: str,
        llm_input: TinyLLMInput,
        result: Any,
        kwargs: dict[str, Any],
    ) -> None:
        start_time = self.call_start_times.pop(run_id, None)
        if start_time:
            duration = time.time() - start_time
            self.call_durations.append(duration)

            print(
                TinyColorPrinter.custom(
                    'LLM CALL END',
                    f'[Run: {run_id[:8]}...] Duration: {duration:.2f}s',
                    color='GREEN',
                )
            )

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about LLM calls."""
        if not self.call_durations:
            return {'total_calls': 0, 'avg_duration': 0, 'total_duration': 0}

        return {
            'total_calls': self.total_calls,
            'avg_duration': sum(self.call_durations) / len(self.call_durations),
            'total_duration': sum(self.call_durations),
            'min_duration': min(self.call_durations),
            'max_duration': max(self.call_durations),
        }


@register_middleware('tool_audit')
class ToolCallAuditMiddleware(TinyBaseMiddleware):
    """Middleware that audits all tool calls with detailed logging."""

    def __init__(self) -> None:
        self.tool_calls: list[dict[str, Any]] = []
        self.tool_start_times: dict[str, float] = {}

    async def before_tool_call(
        self,
        *,
        run_id: str,
        tool: AbstractTool,
        args: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> None:
        key = f'{run_id}:{tool.info.name}'
        self.tool_start_times[key] = time.time()

        print(
            TinyColorPrinter.custom(
                'TOOL CALL',
                f'[Run: {run_id[:8]}...] Tool: {tool.info.name}\n   Args: {args}',
                color='YELLOW',
            )
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
        key = f'{run_id}:{tool.info.name}'
        start_time = self.tool_start_times.pop(key, None)
        duration = time.time() - start_time if start_time else 0

        audit_entry = {
            'run_id': run_id,
            'tool_name': tool.info.name,
            'args': args,
            'result': str(result)[:100],  # Truncate for audit log
            'duration': duration,
            'timestamp': time.time(),
        }
        self.tool_calls.append(audit_entry)

        print(
            TinyColorPrinter.custom(
                'TOOL RESULT',
                f'[Run: {run_id[:8]}...] Tool: {tool.info.name}\n   Result: {result}\n   Duration: {duration:.3f}s',
                color='MAGENTA',
            )
        )

    async def on_error(
        self, *, run_id: str, e: Exception, kwargs: dict[str, Any]
    ) -> None:
        print(
            TinyColorPrinter.error(
                f'[Run: {run_id[:8]}...] Error during tool execution: {e}'
            )
        )

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return the complete audit log of tool calls."""
        return self.tool_calls


plan_prompt = PlanPromptTemplate(
    init_plan='Plan for solving: {{ task }} with tools: {{ tools }}',
    update_plan=(
        'Update plan for: {{ task }} '
        'using tools: {{ tools }}, '
        'history: {{ history }}, '
        'steps so far: {{ steps }}, '
        'remaining: {{ remaining_steps }}'
    ),
)

action_prompt = ActionPromptTemplate(
    system='You are a helpful agent. Use the available tools to complete tasks.',
    final_answer=(
        'Solve task: {{ task }} using steps {{ steps }}, tools {{ tools }}, '
        'and tool calls {{ tool_calls }}. '
        'Conversation so far: {{ history }}'
    ),
)

fallback_prompt = FallbackAnswerPromptTemplate(
    fallback_answer=(
        'Provide the final answer for {{ task }} '
        'based on history: {{ history }} and steps {{ steps }}'
    )
)

prompt_template = MultiStepPromptTemplate(
    plan=plan_prompt,
    acter=action_prompt,
    fallback=fallback_prompt,
)


def main() -> None:
    plan_middleware = PlanTrackingMiddleware()
    answer_middleware = AnswerLoggingMiddleware()
    timing_middleware = LLMCallTimingMiddleware()
    audit_middleware = ToolCallAuditMiddleware()

    agent = TinyMultiStepAgent(
        llm=build_llm('openai:gpt-4o'),
        prompt_template=prompt_template,
        tools=[greet, add_numbers],
        max_iterations=3,
        middleware=[
            plan_middleware,
            timing_middleware,
            audit_middleware,
            answer_middleware,
        ],
        memory=BufferChatMemory(),
    )

    print('\n' + '=' * 60)
    print('Running agent with 4 custom middlewares:')
    print('\t1. PlanTrackingMiddleware - Tracks planning steps')
    print('\t2. LLMCallTimingMiddleware - Tracks LLM call timing')
    print('\t3. ToolCallAuditMiddleware - Audits tool calls')
    print('\t4. AnswerLoggingMiddleware - Logs final answers')
    print('=' * 60 + '\n')

    result = agent.run('Say hello to Alice and then add 5 and 7')

    print('\n' + '=' * 60)
    print('Agent Result:', result)
    print('=' * 60)

    print('\nLLM Call Statistics:')
    stats = timing_middleware.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f'\t{key}: {value:.3f}s')
        else:
            print(f'\t{key}: {value}')

    print('\nPlanning Steps Collected:')
    for i, plan_step in enumerate(plan_middleware.get_all_plans(), 1):
        print(f'\t{i}. {plan_step}')

    print('\nTool Call Audit Log:')
    for entry in audit_middleware.get_audit_log():
        print(f'\t- {entry["tool_name"]}: {entry["args"]} -> {entry["result"]}')

    print('\nAll Answers Collected:')
    for i, ans in enumerate(answer_middleware.get_all_answers(), 1):
        print(f'\t{i}. {ans[:100]}...' if len(ans) > 100 else f'\t{i}. {ans}')


if __name__ == '__main__':
    main()
