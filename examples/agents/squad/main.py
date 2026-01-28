from pathlib import Path
from typing import Any

from pydantic import Field

from tinygent.agents import TinyMultiStepAgent
from tinygent.agents import TinyReActAgent
from tinygent.agents import TinySquadAgent
from tinygent.agents.middleware import TinyBaseMiddleware
from tinygent.agents.middleware import register_middleware
from tinygent.agents.squad_agent import AgentSquadMember
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.factory import build_llm
from tinygent.core.types import TinyLLMInput
from tinygent.core.types import TinyModel
from tinygent.logging import setup_logger
from tinygent.memory import BufferChatMemory
from tinygent.memory import BufferWindowChatMemory
from tinygent.prompts import MultiStepPromptTemplate
from tinygent.prompts import ReActPromptTemplate
from tinygent.prompts import SquadPromptTemplate
from tinygent.tools import register_reasoning_tool
from tinygent.tools import register_tool
from tinygent.utils import TinyColorPrinter
from tinygent.utils import tiny_yaml_load

logger = setup_logger('debug')


@register_middleware('squad_delegation')
class SquadDelegationMiddleware(TinyBaseMiddleware):
    """Middleware that tracks agent delegation and coordination in Squad agent."""

    def __init__(self) -> None:
        self.delegations: list[dict[str, Any]] = []
        self.llm_calls = 0
        self.tool_calls_count = 0
        self.current_plan: str | None = None

    async def on_plan(self, *, run_id: str, plan: str, kwargs: dict[str, Any]) -> None:
        self.current_plan = plan
        print(
            TinyColorPrinter.custom(
                'SQUAD PLAN',
                f'[Run: {run_id[:8]}...]\\n{plan[:200]}...'
                if len(plan) > 200
                else f'[Run: {run_id[:8]}...]\\n{plan}',
                color='CYAN',
            )
        )

    async def before_llm_call(
        self, *, run_id: str, llm_input: TinyLLMInput, kwargs: dict[str, Any]
    ) -> None:
        self.llm_calls += 1
        print(
            TinyColorPrinter.custom(
                'SQUAD LLM',
                f'[Run: {run_id[:8]}...] Coordinator LLM call #{self.llm_calls}',
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
        self.tool_calls_count += 1
        # Check if this is a delegation to a squad member
        is_delegation = 'agent' in tool.info.name.lower() or any(
            keyword in str(args).lower()
            for keyword in ['weather', 'geographic', 'destination']
        )

        if is_delegation:
            delegation_info = {
                'run_id': run_id,
                'delegated_to': tool.info.name,
                'args': args,
            }
            self.delegations.append(delegation_info)
            print(
                TinyColorPrinter.custom(
                    'DELEGATION',
                    f'[Run: {run_id[:8]}...] Delegating to: {tool.info.name}',
                    color='YELLOW',
                )
            )
        else:
            print(
                TinyColorPrinter.custom(
                    'TOOL CALL',
                    f'[Run: {run_id[:8]}...] Using tool: {tool.info.name}',
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
        result_preview = (
            str(result)[:100] + '...' if len(str(result)) > 100 else str(result)
        )
        print(
            TinyColorPrinter.custom(
                'RESULT',
                f'[Run: {run_id[:8]}...] {tool.info.name} -> {result_preview}',
                color='GREEN',
            )
        )

    async def on_answer(
        self, *, run_id: str, answer: str, kwargs: dict[str, Any]
    ) -> None:
        print(
            TinyColorPrinter.custom(
                'SQUAD ANSWER',
                f'[Run: {run_id[:8]}...] Squad completed with {len(self.delegations)} delegations',
                color='GREEN',
            )
        )

    async def on_error(
        self, *, run_id: str, e: Exception, kwargs: dict[str, Any]
    ) -> None:
        print(TinyColorPrinter.error(f'[Run: {run_id[:8]}...] Squad Error: {e}'))

    def get_summary(self) -> dict[str, Any]:
        """Return summary of squad coordination."""
        return {
            'total_llm_calls': self.llm_calls,
            'total_tool_calls': self.tool_calls_count,
            'total_delegations': len(self.delegations),
            'delegated_agents': list({d['delegated_to'] for d in self.delegations}),
        }

    def get_delegation_log(self) -> list[dict[str, Any]]:
        """Return complete delegation log."""
        return self.delegations


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


class SumInput(TinyModel):
    numbers: list[int] = Field(..., description='A list of numbers to sum.')


@register_tool
def calculate_sum(data: SumInput) -> int:
    """Calculate the sum of a list of numbers."""
    return sum(data.numbers)


def main():
    squad_middleware = SquadDelegationMiddleware()

    squad_agent = TinySquadAgent(
        llm=build_llm('openai:gpt-4o', temperature=0.1),
        prompt_template=SquadPromptTemplate(
            **tiny_yaml_load(str(Path(__file__).parent / 'prompts.yaml'))
        ),
        squad=[
            AgentSquadMember(
                name='weather_agent',
                description='An agent that provides weather information.',
                agent=TinyReActAgent(
                    llm=build_llm('openai:gpt-4o', temperature=0.1),
                    max_iterations=3,
                    memory=BufferChatMemory(),
                    tools=[get_weather],
                    prompt_template=ReActPromptTemplate(
                        **tiny_yaml_load(
                            str(Path(__file__).parent.parent / 'react' / 'prompts.yaml')
                        )
                    ),
                ),
            ),
            AgentSquadMember(
                name='geoghraphic_agent',
                description='An agent that provides geographic information.',
                agent=TinyMultiStepAgent(
                    llm=build_llm('openai:gpt-4o', temperature=0.1),
                    memory=BufferWindowChatMemory(k=3),
                    tools=[get_best_destination],
                    prompt_template=MultiStepPromptTemplate(
                        **tiny_yaml_load(
                            str(
                                Path(__file__).parent.parent
                                / 'multi-step'
                                / 'prompts.yaml'
                            )
                        )
                    ),
                ),
            ),
        ],
        memory=BufferChatMemory(),
        tools=[calculate_sum],
        middleware=[squad_middleware],
    )

    result = squad_agent.run(
        'What is the best travel destination and what is the weather like there?'
    )

    logger.info('[RESULT] %s', result)
    logger.info('[MEMORY] %s', squad_agent.memory.load_variables())
    logger.info('[AGENT SUMMARY] %s', str(squad_agent))

    print('\nSquad Delegation Summary:')
    summary = squad_middleware.get_summary()
    for key, value in summary.items():
        print(f'\t{key}: {value}')

    print('\nDelegation Log:')
    for delegation in squad_middleware.get_delegation_log():
        print(f'\t- Delegated to: {delegation["delegated_to"]}')


if __name__ == '__main__':
    main()
