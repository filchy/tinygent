from tinygent.core.types.prompt_template import TinyPromptTemplate


class LLMToolSelectorPromptTemplate(TinyPromptTemplate, TinyPromptTemplate.UserSystem):
    _template_fields = {'user': {'tools'}}
