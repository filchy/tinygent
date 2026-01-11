from pathlib import Path
from typing import Any

from pydantic import Field

from tinygent.agents.middleware.base import AgentMiddleware
from tinygent.agents.middleware.base import register_middleware
from tinygent.agents.react_agent import ReActPromptTemplate
from tinygent.agents.react_agent import TinyReActAgent
from tinygent.datamodels.tool import AbstractTool
from tinygent.factory import build_llm
from tinygent.logging import setup_logger
from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.tools.tool import register_tool
from tinygent.types.base import TinyModel
from tinygent.utils.color_printer import TinyColorPrinter
from tinygent.utils.yaml import tiny_yaml_load

logger = setup_logger('debug')


@register_middleware('react_cycle')
class ReActCycleMiddleware(AgentMiddleware):
    """Middleware that tracks the Thought-Action-Observation cycle in ReAct agent."""

    def __init__(self) -> None:
        self.cycles: list[dict[str, Any]] = []
        self.current_cycle: dict[str, Any] = {}
        self.iteration = 0

    def on_reasoning(self, *, run_id: str, reasoning: str) -> None:
        self.iteration += 1
        self.current_cycle = {
            'iteration': self.iteration,
            'thought': reasoning,
        }
        print(
            TinyColorPrinter.custom(
                'THOUGHT',
                f'[Iteration #{self.iteration}] {reasoning[:150]}...'
                if len(reasoning) > 150
                else f'[Iteration #{self.iteration}] {reasoning}',
                color='CYAN',
            )
        )

    def before_tool_call(
        self, *, run_id: str, tool: AbstractTool, args: dict[str, Any]
    ) -> None:
        self.current_cycle['action'] = {
            'tool': tool.info.name,
            'args': args,
        }
        print(
            TinyColorPrinter.custom(
                'ACTION',
                f'[Iteration #{self.iteration}] {tool.info.name}({args})',
                color='YELLOW',
            )
        )

    def after_tool_call(
        self,
        *,
        run_id: str,
        tool: AbstractTool,
        args: dict[str, Any],
        result: Any,
    ) -> None:
        self.current_cycle['observation'] = str(result)
        self.cycles.append(self.current_cycle)
        print(
            TinyColorPrinter.custom(
                'OBSERVATION',
                f'[Iteration #{self.iteration}] {str(result)[:100]}...'
                if len(str(result)) > 100
                else f'[Iteration #{self.iteration}] {result}',
                color='MAGENTA',
            )
        )

    def on_answer(self, *, run_id: str, answer: str) -> None:
        print(
            TinyColorPrinter.custom(
                'FINAL ANSWER',
                f'[Run: {run_id[:8]}...] After {self.iteration} iterations',
                color='GREEN',
            )
        )

    def on_answer_chunk(self, *, run_id: str, chunk: str, idx: str) -> None:
        # For streaming responses
        print(
            TinyColorPrinter.custom(
                'STREAM',
                f'[{idx}] {chunk}',
                color='BLUE',
            )
        )

    def on_error(self, *, run_id: str, e: Exception) -> None:
        print(
            TinyColorPrinter.error(
                f'[Iteration #{self.iteration}] Error: {e}'
            )
        )

    def get_cycle_log(self) -> list[dict[str, Any]]:
        """Return the complete cycle log."""
        return self.cycles

    def get_summary(self) -> dict[str, Any]:
        """Return summary of ReAct cycles."""
        tools_used = [c.get('action', {}).get('tool') for c in self.cycles if 'action' in c]
        return {
            'total_iterations': self.iteration,
            'completed_cycles': len(self.cycles),
            'tools_used': list(set(tools_used)),
        }


# NOTE: Using @register_tool decorator to register tools globally,
# allowing them to be discovered and reused by:
# - quick.py via discover_and_register_components()
# - CLI terminal command via config-based agent building


class WeatherInput(TinyModel):
    location: str = Field(..., description='The location to get the weather for.')


@register_tool
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


async def main():
    react_agent_prompt = tiny_yaml_load(str(Path(__file__).parent / 'prompts.yaml'))

    react_middleware = ReActCycleMiddleware()

    react_agent = TinyReActAgent(
        llm=build_llm('openai:gpt-4o', temperature=0.1),
        max_iterations=3,
        memory=BufferChatMemory(),
        prompt_template=ReActPromptTemplate(**react_agent_prompt),
        tools=[get_weather, get_best_destination],
        middleware=[react_middleware],
    )

    result: str = ''
    async for chunk in react_agent.run_stream(
        'What is the best travel destination and what is the weather like there?'
    ):
        logger.info('[STREAM CHUNK] %s', chunk)
        result += chunk

    logger.info('[RESULT] %s', result)
    logger.info('[MEMORY] %s', react_agent.memory.load_variables())
    logger.info('[AGENT SUMMARY] %s', str(react_agent))

    print('\nReAct Cycle Summary:')
    summary = react_middleware.get_summary()
    for key, value in summary.items():
        print(f'\t{key}: {value}')

    print('\nCycle Log:')
    for cycle in react_middleware.get_cycle_log():
        print(f'\tIteration {cycle.get("iteration")}:')
        print(f'\t\tThought: {cycle.get("thought", "N/A")[:80]}...')
        if 'action' in cycle:
            print(f'\t\tAction: {cycle["action"]["tool"]}')
        if 'observation' in cycle:
            print(f'\t\tObservation: {cycle["observation"][:50]}...')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
