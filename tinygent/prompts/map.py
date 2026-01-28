"""MAP agent prompt templates."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinygent.core.prompts.agents.factory.map_agent import get_prompt_template
    from tinygent.core.prompts.agents.template.map_agent import MapPromptTemplate

__all__ = ['MapPromptTemplate', 'get_prompt_template']


def __getattr__(name):
    if name == 'MapPromptTemplate':
        from tinygent.core.prompts.agents.template.map_agent import MapPromptTemplate

        return MapPromptTemplate

    if name == 'get_prompt_template':
        from tinygent.core.prompts.agents.factory.map_agent import get_prompt_template

        return get_prompt_template

    raise AttributeError(name)
