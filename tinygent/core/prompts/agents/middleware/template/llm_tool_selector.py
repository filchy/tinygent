from tinygent.core.prompt import TinyPrompt


class LLMToolSelectorPromptTemplate(TinyPrompt, TinyPrompt.UserSystem):
    _template_fields = {'user': {'tools'}}
