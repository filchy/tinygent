from tinygent.types.prompt_template import TinyPromptTemplate


class PlanPromptTemplate(TinyPromptTemplate):
    """Used to generate or update the plan."""

    init_plan: str
    update_plan: str

    _template_fields = {
        'init_plan': {'task', 'tools'},
        'update_plan': {'task', 'tools', 'history', 'steps', 'remaining_steps'},
    }


class ActionPromptTemplate(TinyPromptTemplate):
    """Used to generate the final answer or action."""

    system: str
    final_answer: str

    _template_fields = {
        'final_answer': {'task', 'tools', 'history', 'steps', 'tool_calls'},
    }


class FallbackAnswerPromptTemplate(TinyPromptTemplate):
    """Used to generate the final answer if maximum steps achieved."""

    fallback_answer: str

    _template_fields = {
        'fallback_answer': {'task', 'history', 'steps'},
    }


class MultiStepPromptTemplate(TinyPromptTemplate):
    """Prompt templates for the multi-step agent."""

    plan: PlanPromptTemplate
    acter: ActionPromptTemplate
    fallback: FallbackAnswerPromptTemplate
