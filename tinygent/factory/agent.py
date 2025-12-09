from typing import Any
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

        if isinstance(llm, str):
            from tinygent.factory.llm import parse_model

            model_provider, model_name = parse_model(llm, llm_provider)
            llm_config_dict: Any = {
                'type': model_provider,
                'model': model_name,
            }
            if llm_temperature:
                llm_config_dict['temperature'] = llm_temperature

        elif isinstance(llm, AbstractLLMConfig):
            llm_config_dict = llm.model_dump()

        else:
            llm_config_dict = llm

        agent = {'type': agent, 'llm': llm_config_dict}
        if tools:
            from tinygent.factory.tool import build_tool

            agent['tools'] = [
                t if isinstance(t, AbstractTool) else build_tool(t) for t in tools
            ]

        if memory:
            from tinygent.factory.memory import build_memory

            agent['memory'] = (
                memory if isinstance(memory, AbstractMemory) else build_memory(memory)
            )

    if isinstance(agent, AbstractAgentConfig):
        agent = agent.model_dump()

    agent_config = parse_config(
        agent, lambda: GlobalRegistry.get_registry().get_agents()
    )
    return agent_config.build()
