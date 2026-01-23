from tinygent.types.prompt_template import TinyPromptTemplate


class SummaryUpdatePromptTemplate(TinyPromptTemplate, TinyPromptTemplate.UserSystem):
    _template_fields = {'user': {'summary', 'new_lines'}}
