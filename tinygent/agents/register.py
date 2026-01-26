from tinygent.agents.map_agent import TinyMAPAgent
from tinygent.agents.map_agent import TinyMAPAgentConfig
from tinygent.agents.multi_step_agent import TinyMultiStepAgent
from tinygent.agents.multi_step_agent import TinyMultiStepAgentConfig
from tinygent.agents.react_agent import TinyReActAgent
from tinygent.agents.react_agent import TinyReActAgentConfig
from tinygent.agents.squad_agent import TinySquadAgent
from tinygent.agents.squad_agent import TinySquadAgentConfig
from tinygent.core.runtime.global_registry import GlobalRegistry


def _register_agents() -> None:
    # register agents
    registry = GlobalRegistry().get_registry()

    registry.register_agent('multistep', TinyMultiStepAgentConfig, TinyMultiStepAgent)
    registry.register_agent('react', TinyReActAgentConfig, TinyReActAgent)
    registry.register_agent('squad', TinySquadAgentConfig, TinySquadAgent)
    registry.register_agent('map', TinyMAPAgentConfig, TinyMAPAgent)


_register_agents()
