from tinygent.core.prompt import TinyPrompt
from tinygent.core.types.base import TinyModel


class ReasonPromptTemplate(TinyPrompt):
    """Used to define the reasoning step."""

    init: str
    update: str

    _template_fields = {'init': {'task'}, 'update': {'task', 'overview'}}


class ActionPromptTemplate(TinyPrompt):
    """Used to define the final answer or action."""

    action: str

    _template_fields = {'action': {'reasoning', 'tools'}}


class FallbackPromptTemplate(TinyPrompt):
    """Used to define the fallback if agent don't answer in time."""

    fallback_answer: str

    _template_fields = {'fallback_answer': {'task', 'overview'}}


class ReActPromptTemplate(TinyModel):
    """Prompt template for ReAct Agent."""

    reason: ReasonPromptTemplate
    action: ActionPromptTemplate
    fallback: FallbackPromptTemplate
