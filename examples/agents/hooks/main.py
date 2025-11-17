from pydantic import Field

from tinygent.agents.multi_step_agent import ActionPromptTemplate
from tinygent.agents.multi_step_agent import FallbackAnswerPromptTemplate
from tinygent.agents.multi_step_agent import MultiStepPromptTemplate
from tinygent.agents.multi_step_agent import PlanPromptTemplate
from tinygent.agents.multi_step_agent import TinyMultiStepAgent
from tinygent.llms.base import init_llm
from tinygent.tools import reasoning_tool
from tinygent.types.base import TinyModel
from tinygent.utils.color_printer import TinyColorPrinter


class GreetInput(TinyModel):
    name: str = Field(..., description='The name of the person to greet.')


@reasoning_tool
def greet(data: GreetInput) -> str:
    """Return a simple greeting."""
    return f'Hello, {data.name}!'


def before_llm(inp):
    print(TinyColorPrinter.custom('BEFORE LLM', f'Input: {inp}', color='CYAN'))


def after_llm(inp, result):
    print(TinyColorPrinter.custom('AFTER LLM', f'Result: {result}', color='GREEN'))


def before_tool(tool, args):
    print(
        TinyColorPrinter.custom(
            'TOOL CALL', f'{tool.info.name} with args={args}', color='YELLOW'
        )
    )


def after_tool(tool, args, result):
    print(
        TinyColorPrinter.custom(
            'TOOL RESULT', f'{tool.info.name} → {result}', color='MAGENTA'
        )
    )


def reasoning_hook(r):
    print(TinyColorPrinter.custom('REASONING', r, color='BLUE'))


def tool_reasoning_hook(r):
    print(TinyColorPrinter.custom('TOOL REASONING', r, color='CYAN'))


def answer_hook(ans):
    print(TinyColorPrinter.custom('FINAL ANSWER', ans, color='GREEN'))


def error_hook(e):
    print(TinyColorPrinter.error(f'Error: {e}'))


plan_prompt = PlanPromptTemplate(
    init_plan='Plan for solving: {{ task }} with tools: {{ tools }}',
    update_plan=(
        'Update plan for: {{ task }} '
        'using tools: {{ tools }}, '
        'history: {{ history }}, '
        'steps so far: {{ steps }}, '
        'remaining: {{ remaining_steps }}'
    ),
)

action_prompt = ActionPromptTemplate(
    system='You are a helpful agent.',
    final_answer=(
        'Solve task: {{ task }} using steps {{ steps }}, tools {{ tools }}, '
        'and tool calls {{ tool_calls }}. '
        'Conversation so far: {{ history }}'
    ),
)

fallback_prompt = FallbackAnswerPromptTemplate(
    fallback_answer=(
        'Provide the final answer for {{ task }} '
        'based on history: {{ history }} and steps {{ steps }}'
    )
)

prompt_template = MultiStepPromptTemplate(
    plan=plan_prompt,
    acter=action_prompt,
    fallback=fallback_prompt,
)


def main():
    agent = TinyMultiStepAgent(
        llm=init_llm('openai:gpt-4o'),
        prompt_template=prompt_template,
        tools=[greet],
        max_iterations=3,
        on_before_llm_call=before_llm,
        on_after_llm_call=after_llm,
        on_before_tool_call=before_tool,
        on_after_tool_call=after_tool,
        on_tool_reasoning=tool_reasoning_hook,
        on_reasoning=reasoning_hook,
        on_answer=answer_hook,
        on_error=error_hook,
    )

    result = agent.run('Say hello to Alice')
    print('Result from agent:', result)


if __name__ == '__main__':
    main()
