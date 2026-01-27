from tinygent.core.prompt import TinyPrompt


class LLMCrossEncoderPromptTemplate(TinyPrompt):
    """Prompt template for LLM Cross-encoder."""

    ranking: TinyPrompt.UserSystem

    _template_fields = {
        'ranking.user': {'query', 'text', 'min_range_val', 'max_range_val'}
    }
