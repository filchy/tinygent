from collections.abc import Sequence
import logging
from typing import overload

from tinygent.core.datamodels.agent import AbstractAgent
from tinygent.core.datamodels.agent import AbstractAgentConfig
from tinygent.core.datamodels.llm import AbstractLLM
from tinygent.core.datamodels.llm import AbstractLLMConfig
from tinygent.core.datamodels.memory import AbstractMemory
from tinygent.core.datamodels.memory import AbstractMemoryConfig
from tinygent.core.datamodels.middleware import AbstractMiddleware
from tinygent.core.datamodels.middleware import AbstractMiddlewareConfig
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.datamodels.tool import AbstractToolConfig
from tinygent.core.factory.helper import check_modules
from tinygent.core.factory.helper import parse_config
from tinygent.core.runtime.global_registry import GlobalRegistry

logger = logging.getLogger(__name__)


@overload
def build_agent(
    agent: dict | AbstractAgentConfig,
) -> AbstractAgent: ...


@overload
def build_agent(
    agent: dict | AbstractAgentConfig,
    *,
    middleware: Sequence[AbstractMiddleware | AbstractMiddlewareConfig | str] = [],
    llm: dict | AbstractLLM | AbstractLLMConfig | str | None = None,
    tools: list[dict | AbstractTool | AbstractToolConfig | str] | None = None,
    memory: dict | AbstractMemory | AbstractMemoryConfig | str | None = None,
) -> AbstractAgent: ...


@overload
def build_agent(
    agent: str,
    *,
    middleware: Sequence[AbstractMiddleware | AbstractMiddlewareConfig | str] = [],
    llm: dict | AbstractLLM | AbstractLLMConfig | str | None = None,
    llm_provider: str | None = None,
    llm_temperature: float | None = None,
    tools: list[dict | AbstractTool | AbstractToolConfig | str] | None = None,
    memory: dict | AbstractMemory | AbstractMemoryConfig | str | None = None,
) -> AbstractAgent: ...


def build_agent(
    agent: dict | AbstractAgentConfig | str,
    *,
    middleware: Sequence[AbstractMiddleware | AbstractMiddlewareConfig | str] = [],
    llm: dict | AbstractLLM | AbstractLLMConfig | str | None = None,
    llm_provider: str | None = None,
    llm_temperature: float | None = None,
    tools: list[dict | AbstractTool | AbstractToolConfig | str] | None = None,
    memory: dict | AbstractMemory | AbstractMemoryConfig | str | None = None,
) -> AbstractAgent:
    """Build tiny agent."""
    check_modules()

    if isinstance(agent, str):
        if llm is None:
            raise ValueError(
                f'When building agent by name ("{agent}"), you must provide atleast the "llm" parameter!'
            )

        agent = {'type': agent}

    if isinstance(agent, AbstractAgentConfig):
        agent = agent.model_dump()

    if llm:
        from tinygent.core.factory.llm import build_llm

        if agent.get('llm'):
            logger.warning('Overwriting existing agents llm with new one.')

        agent['llm'] = (
            llm
            if isinstance(llm, AbstractLLM)
            else build_llm(llm, provider=llm_provider, temperature=llm_temperature)
        )

    if tools:
        from tinygent.core.factory.tool import build_tool

        if agent.get('tools'):
            logger.warning('Overwriting existing agents tools with new ones.')

        agent['tools'] = [
            t if isinstance(t, AbstractTool) else build_tool(t) for t in tools
        ]

    if memory:
        from tinygent.core.factory.memory import build_memory

        if agent.get('memory'):
            logger.warning('Overwriting existing agents memory with new one.')

        agent['memory'] = (
            memory if isinstance(memory, AbstractMemory) else build_memory(memory)
        )

    if selected := (middleware if middleware else agent.get('middleware', [])):
        from tinygent.core.factory.middleware import build_middleware

        agent['middleware'] = [
            m if isinstance(m, AbstractMiddleware) else build_middleware(m)
            for m in selected
        ]

    agent_config = parse_config(
        agent, lambda: GlobalRegistry.get_registry().get_agents()
    )

    return agent_config.build()
