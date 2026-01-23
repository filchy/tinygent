from tinygent.core.types.prompt_template import TinyPromptTemplate


class LLMCrossEncoderPromptTemplate(TinyPromptTemplate):
    """Prompt template for LLM Cross-encoder."""

    ranking: TinyPromptTemplate.UserSystem

    _template_fields = {
        'ranking.user': {'query', 'text', 'min_range_val', 'max_range_val'}
    }
