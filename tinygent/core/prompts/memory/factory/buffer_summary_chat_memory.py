from tinygent.core.prompts.memory.template.buffer_summary_chat_memory import (
    SummaryUpdatePromptTemplate,
)


def get_prompt_template() -> SummaryUpdatePromptTemplate:
    return SummaryUpdatePromptTemplate(
        system=(
            'Reason step-by-step internally and keep the reasoning hidden. '
            'Return only the final summary.'
        ),
        user="""
You are a summarization assistant.

Use private, step-by-step reasoning to update the summary, but DO NOT reveal your reasoning.
Only output the final updated summary.

Current summary:
{{ summary }}

New lines of conversation:
{{ new_lines }}

Updated summary (concise, factual, no meta commentary):
        """,
    )
