from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinygent.prompts.map import MapPromptTemplate
    from tinygent.prompts.multistep import MultiStepPromptTemplate
    from tinygent.prompts.react import ReActPromptTemplate
    from tinygent.prompts.squad import SquadPromptTemplate

    from .base_agent import TinyBaseAgent
    from .base_agent import TinyBaseAgentConfig
    from .map_agent import TinyMAPAgent
    from .map_agent import TinyMAPAgentConfig
    from .multi_step_agent import TinyMultiStepAgent
    from .multi_step_agent import TinyMultiStepAgentConfig
    from .react_agent import TinyReActAgent
    from .react_agent import TinyReActAgentConfig
    from .squad_agent import TinySquadAgent
    from .squad_agent import TinySquadAgentConfig

__all__ = [
    'TinyBaseAgent',
    'TinyBaseAgentConfig',
    'TinyMultiStepAgent',
    'TinyMultiStepAgentConfig',
    'TinyReActAgent',
    'TinyReActAgentConfig',
    'TinySquadAgent',
    'TinySquadAgentConfig',
    'TinyMAPAgent',
    'TinyMAPAgentConfig',
    'ReActPromptTemplate',
    'MultiStepPromptTemplate',
    'SquadPromptTemplate',
    'MapPromptTemplate',
]


def __getattr__(name):
    if name == 'TinyBaseAgent':
        from .base_agent import TinyBaseAgent

        return TinyBaseAgent

    if name == 'TinyBaseAgentConfig':
        from .base_agent import TinyBaseAgentConfig

        return TinyBaseAgentConfig

    if name == 'TinyMultiStepAgent':
        from .multi_step_agent import TinyMultiStepAgent

        return TinyMultiStepAgent

    if name == 'TinyMultiStepAgentConfig':
        from .multi_step_agent import TinyMultiStepAgentConfig

        return TinyMultiStepAgentConfig

    if name == 'TinyReActAgent':
        from .react_agent import TinyReActAgent

        return TinyReActAgent

    if name == 'TinyReActAgentConfig':
        from .react_agent import TinyReActAgentConfig

        return TinyReActAgentConfig

    if name == 'TinySquadAgent':
        from .squad_agent import TinySquadAgent

        return TinySquadAgent

    if name == 'TinySquadAgentConfig':
        from .squad_agent import TinySquadAgentConfig

        return TinySquadAgentConfig

    if name == 'TinyMAPAgent':
        from .map_agent import TinyMAPAgent

        return TinyMAPAgent

    if name == 'TinyMAPAgentConfig':
        from .map_agent import TinyMAPAgentConfig

        return TinyMAPAgentConfig

    if name == 'ReActPromptTemplate':
        from tinygent.prompts.react import ReActPromptTemplate

        return ReActPromptTemplate

    if name == 'MultiStepPromptTemplate':
        from tinygent.prompts.multistep import MultiStepPromptTemplate

        return MultiStepPromptTemplate

    if name == 'SquadPromptTemplate':
        from tinygent.prompts.squad import SquadPromptTemplate

        return SquadPromptTemplate

    if name == 'MapPromptTemplate':
        from tinygent.prompts.map import MapPromptTemplate

        return MapPromptTemplate

    raise AttributeError(name)
