from tinygent.agents.multi_step_agent import TinyMultiStepAgent
from tinygent.agents.multi_step_agent import TinyMultiStepAgentConfig
from tinygent.runtime.global_registry import GlobalRegistry

GlobalRegistry().get_registry().register_agent(
    'multistep', TinyMultiStepAgentConfig, TinyMultiStepAgent
)
