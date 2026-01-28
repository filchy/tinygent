"""Middleware prompt templates."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinygent.core.prompts.agents.middleware.factory.llm_tool_selector import (
        get_prompt_template as get_llm_tool_selector_prompt_template,
    )
    from tinygent.core.prompts.agents.middleware.template.llm_tool_selector import (
        LLMToolSelectorPromptTemplate,
    )

__all__ = ['LLMToolSelectorPromptTemplate', 'get_llm_tool_selector_prompt_template']


def __getattr__(name):
    if name == 'LLMToolSelectorPromptTemplate':
        from tinygent.core.prompts.agents.middleware.template.llm_tool_selector import (
            LLMToolSelectorPromptTemplate,
        )

        return LLMToolSelectorPromptTemplate

    if name == 'get_llm_tool_selector_prompt_template':
        from tinygent.core.prompts.agents.middleware.factory.llm_tool_selector import (
            get_prompt_template as get_llm_tool_selector_prompt_template,
        )

        return get_llm_tool_selector_prompt_template

    raise AttributeError(name)
