from tinygent.core.types.base import TinyModel
from tinygent.core.types.prompt_template import TinyPromptTemplate


class ClassifierPromptTemplate(TinyPromptTemplate):
    """Used to define the classifier (orchestrator) prompt template."""

    prompt: str

    _template_fields = {'prompt': {'task', 'tools', 'squad_members'}}


class SquadPromptTemplate(TinyModel):
    """Used to define the squad member prompt template."""

    classifier: ClassifierPromptTemplate
