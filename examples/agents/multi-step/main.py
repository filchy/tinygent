from pathlib import Path
from typing import Any

from pydantic import Field

from tinygent.agents import TinyMultiStepAgent
from tinygent.agents.middleware import TinyBaseMiddleware
from tinygent.agents.middleware import register_middleware
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.factory import build_llm
from tinygent.core.types import TinyLLMInput
from tinygent.core.types import TinyModel
from tinygent.logging import setup_logger
from tinygent.memory import BufferChatMemory
from tinygent.memory import BufferWindowChatMemory
from tinygent.memory import CombinedMemory
from tinygent.prompts import MultiStepPromptTemplate
from tinygent.tools import register_reasoning_tool
from tinygent.tools import register_tool
from tinygent.utils import TinyColorPrinter
from tinygent.utils import tiny_yaml_load

logger = setup_logger('debug')


@register_middleware('step_counter')
class StepCounterMiddleware(TinyBaseMiddleware):
    """Middleware that tracks steps and iterations in multi-step agent."""

    def __init__(self) -> None:
        self.current_step = 0
        self.tool_calls: list[dict[str, Any]] = []
        self.plans: list[str] = []

    async def on_plan(self, *, run_id: str, plan: str, kwargs: dict[str, Any]) -> None:
        self.plans.append(plan)
        print(
            TinyColorPrinter.custom(
                'STEP PLAN',
                f'[Run: {run_id[:8]}...] Plan #{len(self.plans)}:\\n{plan[:200]}...'
                if len(plan) > 200
                else f'[Run: {run_id[:8]}...] Plan #{len(self.plans)}:\\n{plan}',
                color='CYAN',
            )
        )

    async def before_llm_call(
        self, *, run_id: str, llm_input: TinyLLMInput, kwargs: dict[str, Any]
    ) -> None:
        self.current_step += 1
        print(
            TinyColorPrinter.custom(
                'STEP',
                f'[Run: {run_id[:8]}...] Starting step #{self.current_step}',
                color='BLUE',
            )
        )

    async def before_tool_call(
        self,
        *,
        run_id: str,
        tool: AbstractTool,
        args: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> None:
        print(
            TinyColorPrinter.custom(
                'TOOL EXECUTION',
                f'[Step #{self.current_step}] Calling: {tool.info.name}',
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
        self.tool_calls.append(
            {
                'step': self.current_step,
                'tool': tool.info.name,
                'args': args,
                'result': str(result)[:100],
            }
        )
        print(
            TinyColorPrinter.custom(
                'TOOL COMPLETE',
                f'[Step #{self.current_step}] {tool.info.name} -> {str(result)[:50]}...',
                color='GREEN',
            )
        )

    async def on_answer(
        self, *, run_id: str, answer: str, kwargs: dict[str, Any]
    ) -> None:
        print(
            TinyColorPrinter.custom(
                'FINAL ANSWER',
                f'[Run: {run_id[:8]}...] Completed in {self.current_step} steps',
                color='GREEN',
            )
        )

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about the multi-step execution."""
        return {
            'total_steps': self.current_step,
            'total_plans': len(self.plans),
            'tool_calls': len(self.tool_calls),
            'tools_used': list({tc['tool'] for tc in self.tool_calls}),
        }


# NOTE: Using @register_tool & @register_reasoning_tool decorator to register tools globally,
# allowing them to be discovered and reused by:
# - quick.py via discover_and_register_components()
# - CLI terminal command via config-based agent building


class WeatherInput(TinyModel):
    location: str = Field(..., description='The location to get the weather for.')


@register_reasoning_tool(
    reasoning_prompt='Provide reasoning for why the weather information is needed.'
)
def get_weather(data: WeatherInput) -> str:
    """Get the current weather in a given location."""

    return f'The weather in {data.location} is sunny with a high of 75°F.'


class GetBestDestinationInput(TinyModel):
    top_k: int = Field(..., description='The number of top destinations to return.')


@register_tool
def get_best_destination(data: GetBestDestinationInput) -> list[str]:
    """Get the best travel destinations."""
    destinations = {'Paris', 'New York', 'Tokyo', 'Barcelona', 'Rome'}
    return list(destinations)[: data.top_k]


def main():
    multi_step_agent_prompt = tiny_yaml_load(str(Path(__file__).parent / 'prompts.yaml'))

    step_middleware = StepCounterMiddleware()

    multi_step_agent = TinyMultiStepAgent(
        llm=build_llm('openai:gpt-4o', temperature=0.1),
        prompt_template=MultiStepPromptTemplate(**multi_step_agent_prompt),
        memory=CombinedMemory(
            memory_list=[
                BufferChatMemory(),
                BufferWindowChatMemory(k=3),
            ]
        ),
        tools=[get_weather, get_best_destination],
        middleware=[step_middleware],
    )

    result = multi_step_agent.run(
        'What is the best travel destination and what is the weather like there?'
    )

    logger.info('[RESULT] %s', result)
    logger.info('[MEMORY] %s', multi_step_agent.memory.load_variables())
    logger.info('[AGENT SUMMARY] %s', str(multi_step_agent))

    print('\nStep Counter Summary:')
    stats = step_middleware.get_stats()
    for key, value in stats.items():
        print(f'\t{key}: {value}')


if __name__ == '__main__':
    main()
