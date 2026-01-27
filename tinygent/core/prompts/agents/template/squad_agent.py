from tinygent.core.prompt import TinyPrompt
from tinygent.core.types.base import TinyModel


class ClassifierPromptTemplate(TinyPrompt):
    """Used to define the classifier (orchestrator) prompt template."""

    prompt: str

    _template_fields = {'prompt': {'task', 'tools', 'squad_members'}}


class SquadPromptTemplate(TinyModel):
    """Used to define the squad member prompt template."""

    classifier: ClassifierPromptTemplate
