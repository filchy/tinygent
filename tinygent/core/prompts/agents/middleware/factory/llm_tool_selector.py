from tinygent.core.prompts.agents.middleware.template.llm_tool_selector import (
    LLMToolSelectorPromptTemplate,
)


def get_prompt_template() -> LLMToolSelectorPromptTemplate:
    return LLMToolSelectorPromptTemplate(
        system=(
            "Your task is to select the most relevant tools for answering the user's query.\n"
            'You will be given a list of available tools with names and descriptions.\n'
            "Return ONLY a JSON object with a single key 'tools' containing a list of tool names.\n"
            'Order the tool names by relevance (most relevant first).\n'
            "Select only tools that are truly useful for solving the user's request.\n"
            'Do NOT invent tool names.\n'
            'Do NOT include explanations or additional text.\n'
            'If multiple tools are useful, include all relevant ones in ranked order.'
        ),
        user='',
    )
