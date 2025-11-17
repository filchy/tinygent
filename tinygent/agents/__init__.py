__all__ = [
    'TinyBaseAgent',
    'TinyBaseAgentConfig',
    'TinyMultiStepAgent',
    'TinyMultiStepAgentConfig',
    'TinyReActAgent',
    'TinyReActAgentConfig',
    'TinySquadAgent',
    'TinySquadAgentConfig',
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

    raise AttributeError(name)
