"""ReAct agent prompt templates."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinygent.core.prompts.agents.factory.react_agent import get_prompt_template
    from tinygent.core.prompts.agents.template.react_agent import ActionPromptTemplate
    from tinygent.core.prompts.agents.template.react_agent import FallbackPromptTemplate
    from tinygent.core.prompts.agents.template.react_agent import ReActPromptTemplate
    from tinygent.core.prompts.agents.template.react_agent import ReasonPromptTemplate

__all__ = [
    'ReActPromptTemplate',
    'ActionPromptTemplate',
    'FallbackPromptTemplate',
    'ReasonPromptTemplate',
    'get_prompt_template',
]


def __getattr__(name):
    if name == 'ReActPromptTemplate':
        from tinygent.core.prompts.agents.template.react_agent import ReActPromptTemplate

        return ReActPromptTemplate

    if name == 'ActionPromptTemplate':
        from tinygent.core.prompts.agents.template.react_agent import (
            ActionPromptTemplate,
        )

        return ActionPromptTemplate

    if name == 'FallbackPromptTemplate':
        from tinygent.core.prompts.agents.template.react_agent import (
            FallbackPromptTemplate,
        )

        return FallbackPromptTemplate

    if name == 'ReasonPromptTemplate':
        from tinygent.core.prompts.agents.template.react_agent import (
            ReasonPromptTemplate,
        )

        return ReasonPromptTemplate

    if name == 'get_prompt_template':
        from tinygent.core.prompts.agents.factory.react_agent import get_prompt_template

        return get_prompt_template

    raise AttributeError(name)
