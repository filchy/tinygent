import logging
from typing import overload

from tinygent.datamodels.agent import AbstractAgent
from tinygent.datamodels.agent import AbstractAgentConfig
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm import AbstractLLMConfig
from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.memory import AbstractMemoryConfig
from tinygent.datamodels.tool import AbstractTool
from tinygent.datamodels.tool import AbstractToolConfig
from tinygent.factory.helper import check_modules
from tinygent.factory.helper import parse_config
from tinygent.runtime.global_registry import GlobalRegistry

logger = logging.getLogger(__name__)


@overload
def build_agent(
    agent: dict | AbstractAgentConfig,
) -> AbstractAgent: ...


@overload
def build_agent(
    agent: dict | AbstractAgentConfig,
    *,
    llm: dict | AbstractLLM | AbstractLLMConfig | str | None = None,
    tools: list[dict | AbstractTool | AbstractToolConfig | str] | None = None,
    memory: dict | AbstractMemory | AbstractMemoryConfig | str | None = None,
) -> AbstractAgent: ...


@overload
def build_agent(
    agent: str,
    *,
    llm: dict | AbstractLLM | AbstractLLMConfig | str | None = None,
    llm_provider: str | None = None,
    llm_temperature: float | None = None,
    tools: list[dict | AbstractTool | AbstractToolConfig | str] | None = None,
    memory: dict | AbstractMemory | AbstractMemoryConfig | str | None = None,
) -> AbstractAgent: ...


def build_agent(
    agent: dict | AbstractAgentConfig | str,
    *,
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
        from tinygent.factory.llm import build_llm

        if agent.get('llm'):
            logger.warning('Overwriting existing agents llm with new one.')

        agent['llm'] = (
            llm
            if isinstance(llm, AbstractLLM)
            else build_llm(llm, provider=llm_provider, temperature=llm_temperature)
        )

    if tools:
        from tinygent.factory.tool import build_tool

        if agent.get('tools'):
            logger.warning('Overwriting existing agents tools with new ones.')

        agent['tools'] = [
            t if isinstance(t, AbstractTool) else build_tool(t) for t in tools
        ]

    if memory:
        from tinygent.factory.memory import build_memory

        if agent.get('memory'):
            logger.warning('Overwriting existing agents memory with new one.')

        agent['memory'] = (
            memory if isinstance(memory, AbstractMemory) else build_memory(memory)
        )

    agent_config = parse_config(
        agent, lambda: GlobalRegistry.get_registry().get_agents()
    )

    return agent_config.build()
