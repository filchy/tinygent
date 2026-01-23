from tinygent.core.types.base import TinyModel
from tinygent.core.types.prompt_template import TinyPromptTemplate


class OrchestratorPromptTemplate(TinyPromptTemplate, TinyPromptTemplate.UserSystem):
    """Used to define orchestrator prompt template."""

    _template_fields = {'user': {'question', 'answer'}}


class MonitorPrompTemplate(TinyPromptTemplate):
    """Used to define monitor prompt template."""

    init: TinyPromptTemplate.UserSystem

    continuos: TinyPromptTemplate.UserSystem

    _template_fields = {
        'init.user': {'question', 'answer'},
        'continuos.user': {'question', 'answer', 'previous_questions'},
    }


class ActorPromptTemplate(TinyPromptTemplate):
    """Used to define actor prompt template."""

    init: TinyPromptTemplate.UserSystem

    init_fixer: TinyPromptTemplate.UserSystem

    continuos: TinyPromptTemplate.UserSystem

    continuos_fixer: TinyPromptTemplate.UserSystem

    evaluator: TinyPromptTemplate.UserSystem

    _template_fields = {
        'init.user': {'question'},
        'init_fixer.user': {'question', 'validation'},
        'continuos.user': {'question', 'previous_questions'},
        'continuos_fixer.user': {'question', 'validation'},
        'evaluator.user': {'state', 'subgoal'},
    }


class ActionProposalPromptTemplate(TinyModel):
    """Used to define action proposal module prompt template."""

    actor: ActorPromptTemplate

    monitor: MonitorPrompTemplate


class TaskDecomposerPromptTemplate(TinyPromptTemplate, TinyPromptTemplate.UserSystem):
    """Used to define task decomposer prompt template."""

    _template_fields = {'user': {'question', 'max_subquestions'}}


class PredictorPromptTemplate(TinyPromptTemplate, TinyPromptTemplate.UserSystem):
    """Used to define predictor prompt template."""

    _template_fields = {'user': {'state', 'proposed_action'}}


class MapPromptTemplate(TinyModel):
    """Prompt template for MAP Agent."""

    task_decomposer: TaskDecomposerPromptTemplate

    action_proposal: ActionProposalPromptTemplate

    predictor: PredictorPromptTemplate

    orchestrator: OrchestratorPromptTemplate
