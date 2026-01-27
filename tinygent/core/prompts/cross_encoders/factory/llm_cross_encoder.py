from tinygent.core.prompt import TinyPrompt
from tinygent.core.prompts.cross_encoders.template.llm_cross_encoder import (
    LLMCrossEncoderPromptTemplate,
)


def get_prompt_template() -> LLMCrossEncoderPromptTemplate:
    return LLMCrossEncoderPromptTemplate(
        ranking=TinyPrompt.UserSystem(
            system="""You are a relevance scoring model acting as a cross-encoder.

Your task is to evaluate how relevant a given TEXT is to a given QUERY.
You must consider semantic meaning, factual alignment, and implied intent.

You must produce a structured output that contains a single numeric field named "score".

Rules:
- The value of "score" must be a number.
- The value must lie within the provided range.
- Higher score means higher relevance.
- Do not include explanations, reasoning, or extra fields.
            """,
            user="""QUERY:
{{ query }}

TEXT:
{{ text }}

Evaluate the relevance of TEXT to QUERY.

Scoring rules:
- {{ min_range_val }} = completely irrelevant
- {{ max_range_val }} = perfectly relevant
- Use intermediate values proportionally

Return the result as structured output with a single field:
- score: a number between {{ min_range_val }} and {{ max_range_val }}
            """,
        )
    )
