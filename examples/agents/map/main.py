from pathlib import Path
from typing import Any

from tinygent.agents import TinyMAPAgent
from tinygent.agents.middleware import TinyBaseMiddleware
from tinygent.agents.middleware import register_middleware
from tinygent.core.factory import build_llm
from tinygent.core.types import TinyLLMInput
from tinygent.logging import setup_logger
from tinygent.memory import BufferChatMemory
from tinygent.prompts import MapPromptTemplate
from tinygent.utils import TinyColorPrinter
from tinygent.utils import tiny_yaml_load

logger = setup_logger('debug')


@register_middleware('plan_progress')
class PlanProgressMiddleware(TinyBaseMiddleware):
    """Middleware that tracks planning progress in MAP agent."""

    def __init__(self) -> None:
        self.plans: list[str] = []
        self.llm_calls = 0

    async def on_plan(self, *, run_id: str, plan: str, kwargs: dict[str, Any]) -> None:
        self.plans.append(plan)
        print(
            TinyColorPrinter.custom(
                'PLAN UPDATE',
                f'[Run: {run_id[:8]}...] Plan #{len(self.plans)}:\n{plan}',
                color='CYAN',
            )
        )

    async def before_llm_call(
        self, *, run_id: str, llm_input: TinyLLMInput, kwargs: dict[str, Any]
    ) -> None:
        self.llm_calls += 1
        print(
            TinyColorPrinter.custom(
                'MAP LLM CALL',
                f'[Run: {run_id[:8]}...] LLM Call #{self.llm_calls}',
                color='BLUE',
            )
        )

    async def on_answer(
        self, *, run_id: str, answer: str, kwargs: dict[str, Any]
    ) -> None:
        print(
            TinyColorPrinter.custom(
                'MAP ANSWER',
                f'[Run: {run_id[:8]}...]\n{answer}',
                color='GREEN',
            )
        )

    async def on_error(
        self, *, run_id: str, e: Exception, kwargs: dict[str, Any]
    ) -> None:
        print(TinyColorPrinter.error(f'[Run: {run_id[:8]}...] MAP Error: {e}'))

    def get_summary(self) -> dict[str, Any]:
        """Return summary of planning progress."""
        return {
            'total_plans': len(self.plans),
            'total_llm_calls': self.llm_calls,
        }


async def main():
    map_agent_prompt = tiny_yaml_load(str(Path(__file__).parent / 'prompts.yaml'))

    plan_middleware = PlanProgressMiddleware()

    agent = TinyMAPAgent(
        llm=build_llm('openai:gpt-4o-mini', temperature=0.1),
        prompt_template=MapPromptTemplate(**map_agent_prompt),
        memory=BufferChatMemory(),
        max_plan_length=3,
        max_branches_per_layer=2,
        max_layer_depth=2,
        max_recurrsion=3,
        middleware=[plan_middleware],
    )

    result = agent.run(
        'Will the Albany in Georgia reach a hundred thousand occupants before the one in New York?'
    )

    logger.info('[RESULT] %s', result)
    logger.info('[MEMORY] %s', agent.memory.load_variables())
    logger.info('[AGENT] %s', str(agent))

    print('\nPlan Progress Summary:')
    summary = plan_middleware.get_summary()
    for key, value in summary.items():
        print(f'\t{key}: {value}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
