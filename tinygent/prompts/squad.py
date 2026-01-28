"""Squad agent prompt templates."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinygent.core.prompts.agents.factory.squad_agent import get_prompt_template
    from tinygent.core.prompts.agents.template.squad_agent import SquadPromptTemplate

__all__ = ['SquadPromptTemplate', 'get_prompt_template']


def __getattr__(name):
    if name == 'SquadPromptTemplate':
        from tinygent.core.prompts.agents.template.squad_agent import SquadPromptTemplate

        return SquadPromptTemplate

    if name == 'get_prompt_template':
        from tinygent.core.prompts.agents.factory.squad_agent import get_prompt_template

        return get_prompt_template

    raise AttributeError(name)
