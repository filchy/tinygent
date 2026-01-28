from pathlib import Path
from typing import Any

from pydantic import Field

from tinygent.agents import TinyReActAgent
from tinygent.agents.middleware import TinyBaseMiddleware
from tinygent.agents.middleware import register_middleware
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.factory import build_llm
from tinygent.core.types import TinyModel
from tinygent.logging import setup_logger
from tinygent.memory import BufferChatMemory
from tinygent.prompts import ReActPromptTemplate
from tinygent.tools import register_tool
from tinygent.utils import TinyColorPrinter
from tinygent.utils import tiny_yaml_load

logger = setup_logger('debug')


@register_middleware('react_tool_tracker')
class ReActToolTrackerMiddleware(TinyBaseMiddleware):
    """Middleware that tracks tool calls in ReAct agent."""

    def __init__(self) -> None:
        self.tool_calls: list[dict[str, Any]] = []
        self.total_calls = 0

    async def before_tool_call(
        self,
        *,
        run_id: str,
        tool: AbstractTool,
        args: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> None:
        self.total_calls += 1
        print(
            TinyColorPrinter.custom(
                'TOOL CALL',
                f'[Call #{self.total_calls}] {tool.info.name}({args})',
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
        call_record = {
            'tool': tool.info.name,
            'args': args,
            'result': str(result),
        }
        self.tool_calls.append(call_record)
        print(
            TinyColorPrinter.custom(
                'TOOL RESULT',
                f'[Call #{self.total_calls}] {str(result)[:100]}...'
                if len(str(result)) > 100
                else f'[Call #{self.total_calls}] {result}',
                color='MAGENTA',
            )
        )

    async def on_tool_reasoning(
        self, *, run_id: str, reasoning: str, kwargs: dict[str, Any]
    ) -> None:
        print(
            TinyColorPrinter.custom(
                'TOOL REASONING',
                f'{reasoning[:150]}...' if len(reasoning) > 150 else reasoning,
                color='CYAN',
            )
        )

    async def on_answer(
        self, *, run_id: str, answer: str, kwargs: dict[str, Any]
    ) -> None:
        print(
            TinyColorPrinter.custom(
                'FINAL ANSWER',
                f'[Run: {run_id[:8]}...] After {self.total_calls} tool calls',
                color='GREEN',
            )
        )

    async def on_answer_chunk(
        self, *, run_id: str, chunk: str, idx: str, kwargs: dict[str, Any]
    ) -> None:
        print(
            TinyColorPrinter.custom(
                'STREAM',
                f'[{idx}] {chunk}',
                color='BLUE',
            )
        )

    async def on_error(
        self, *, run_id: str, e: Exception, kwargs: dict[str, Any]
    ) -> None:
        print(TinyColorPrinter.error(f'Error: {e}'))

    def get_tool_calls(self) -> list[dict[str, Any]]:
        """Return all tool calls."""
        return self.tool_calls

    def get_summary(self) -> dict[str, Any]:
        """Return summary of tool usage."""
        tools_used = [c['tool'] for c in self.tool_calls]
        return {
            'total_calls': self.total_calls,
            'unique_tools': list(set(tools_used)),
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

    react_middleware = ReActToolTrackerMiddleware()

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

    print('\nTool Usage Summary:')
    summary = react_middleware.get_summary()
    for key, value in summary.items():
        print(f'\t{key}: {value}')

    print('\nTool Call Log:')
    for i, call in enumerate(react_middleware.get_tool_calls(), 1):
        print(f'\t{i}. {call["tool"]}({call["args"]}) -> {call["result"][:50]}...')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
