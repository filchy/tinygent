from tinygent.core.prompt import TinyPrompt


class SummaryUpdatePromptTemplate(TinyPrompt, TinyPrompt.UserSystem):
    _template_fields = {'user': {'summary', 'new_lines'}}
