from tinygent.agents.multi_step_agent import TinyMultiStepAgent
from tinygent.agents.multi_step_agent import TinyMultiStepAgentConfig
from tinygent.agents.react_agent import TinyReActAgent
from tinygent.agents.react_agent import TinyReActAgentConfig
from tinygent.runtime.global_registry import GlobalRegistry

_registry = GlobalRegistry().get_registry()

_registry.register_agent('multistep', TinyMultiStepAgentConfig, TinyMultiStepAgent)
_registry.register_agent('react', TinyReActAgentConfig, TinyReActAgent)
