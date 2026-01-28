from tinygent.core.prompt import TinyPrompt
from tinygent.core.types.base import TinyModel


class OrchestratorPromptTemplate(TinyPrompt, TinyPrompt.UserSystem):
    """Used to define orchestrator prompt template."""

    _template_fields = {'user': {'question', 'answer'}}


class MonitorPrompTemplate(TinyPrompt):
    """Used to define monitor prompt template."""

    init: TinyPrompt.UserSystem

    continuos: TinyPrompt.UserSystem

    _template_fields = {
        'init.user': {'question', 'answer'},
        'continuos.user': {'question', 'answer', 'previous_questions'},
    }


class ActorPromptTemplate(TinyPrompt):
    """Used to define actor prompt template."""

    init: TinyPrompt.UserSystem

    init_fixer: TinyPrompt.UserSystem

    continuos: TinyPrompt.UserSystem

    continuos_fixer: TinyPrompt.UserSystem

    evaluator: TinyPrompt.UserSystem

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


class TaskDecomposerPromptTemplate(TinyPrompt, TinyPrompt.UserSystem):
    """Used to define task decomposer prompt template."""

    _template_fields = {'user': {'question', 'max_subquestions'}}


class PredictorPromptTemplate(TinyPrompt, TinyPrompt.UserSystem):
    """Used to define predictor prompt template."""

    _template_fields = {'user': {'state', 'proposed_action'}}


class MapPromptTemplate(TinyModel):
    """Prompt template for MAP Agent."""

    task_decomposer: TaskDecomposerPromptTemplate

    action_proposal: ActionProposalPromptTemplate

    predictor: PredictorPromptTemplate

    orchestrator: OrchestratorPromptTemplate
