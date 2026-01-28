"""MultiStep agent prompt templates."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinygent.core.prompts.agents.factory.multi_agent import get_prompt_template
    from tinygent.core.prompts.agents.template.multi_agent import ActionPromptTemplate
    from tinygent.core.prompts.agents.template.multi_agent import (
        FallbackAnswerPromptTemplate,
    )
    from tinygent.core.prompts.agents.template.multi_agent import MultiStepPromptTemplate
    from tinygent.core.prompts.agents.template.multi_agent import PlanPromptTemplate

__all__ = [
    'MultiStepPromptTemplate',
    'ActionPromptTemplate',
    'FallbackAnswerPromptTemplate',
    'PlanPromptTemplate',
    'get_prompt_template',
]


def __getattr__(name):
    if name == 'MultiStepPromptTemplate':
        from tinygent.core.prompts.agents.template.multi_agent import (
            MultiStepPromptTemplate,
        )

        return MultiStepPromptTemplate

    if name == 'ActionPromptTemplate':
        from tinygent.core.prompts.agents.template.multi_agent import (
            ActionPromptTemplate,
        )

        return ActionPromptTemplate

    if name == 'FallbackAnswerPromptTemplate':
        from tinygent.core.prompts.agents.template.multi_agent import (
            FallbackAnswerPromptTemplate,
        )

        return FallbackAnswerPromptTemplate

    if name == 'PlanPromptTemplate':
        from tinygent.core.prompts.agents.template.multi_agent import PlanPromptTemplate

        return PlanPromptTemplate

    if name == 'get_prompt_template':
        from tinygent.core.prompts.agents.factory.multi_agent import get_prompt_template

        return get_prompt_template

    raise AttributeError(name)
