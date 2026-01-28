"""Prompt templates for Tinygent agents.

Organized by agent type for clean imports:
    from tinygent.prompts import react, multistep, squad, map, middleware

Or import specific templates directly:
    from tinygent.prompts import ReActPromptTemplate, MultiStepPromptTemplate

Or from submodules:
    from tinygent.prompts.react import ReActPromptTemplate
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinygent.core.prompts.agents.template.map_agent import MapPromptTemplate
    from tinygent.core.prompts.agents.template.multi_agent import MultiStepPromptTemplate
    from tinygent.core.prompts.agents.template.react_agent import ReActPromptTemplate
    from tinygent.core.prompts.agents.template.squad_agent import SquadPromptTemplate
    from tinygent.prompts import map
    from tinygent.prompts import middleware
    from tinygent.prompts import multistep
    from tinygent.prompts import react
    from tinygent.prompts import squad

__all__ = [
    'react',
    'multistep',
    'squad',
    'map',
    'middleware',
    'ReActPromptTemplate',
    'MultiStepPromptTemplate',
    'SquadPromptTemplate',
    'MapPromptTemplate',
]


def __getattr__(name):
    if name == 'react':
        from tinygent.prompts import react

        return react

    if name == 'multistep':
        from tinygent.prompts import multistep

        return multistep

    if name == 'squad':
        from tinygent.prompts import squad

        return squad

    if name == 'map':
        from tinygent.prompts import map

        return map

    if name == 'middleware':
        from tinygent.prompts import middleware

        return middleware

    if name == 'ReActPromptTemplate':
        from tinygent.core.prompts.agents.template.react_agent import ReActPromptTemplate

        return ReActPromptTemplate

    if name == 'MultiStepPromptTemplate':
        from tinygent.core.prompts.agents.template.multi_agent import (
            MultiStepPromptTemplate,
        )

        return MultiStepPromptTemplate

    if name == 'SquadPromptTemplate':
        from tinygent.core.prompts.agents.template.squad_agent import SquadPromptTemplate

        return SquadPromptTemplate

    if name == 'MapPromptTemplate':
        from tinygent.core.prompts.agents.template.map_agent import MapPromptTemplate

        return MapPromptTemplate

    raise AttributeError(name)
