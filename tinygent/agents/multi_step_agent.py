from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
import logging
import typing

from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from tinygent.datamodels.agent import AbstractAgent
from tinygent.datamodels.llm_io import TinyLLMInput
from tinygent.datamodels.messages import AllTinyMessages
from tinygent.datamodels.messages import TinyAIMessage
from tinygent.datamodels.messages import TinyChatMessage
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.datamodels.messages import TinyPlanMessage
from tinygent.datamodels.messages import TinyReasoningMessage
from tinygent.datamodels.messages import TinyToolCall
from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.runtime.memory_group import MemoryGroup
from tinygent.tools.default_tools import provide_final_answer
from tinygent.types import TinyModel
from tinygent.utils.answer_validation import is_final_answer
from tinygent.utils.jinja_utils import render_template
from tinygent.utils.jinja_utils import validate_template

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm import AbstractLLM
    from tinygent.datamodels.memory import AbstractMemory
    from tinygent.datamodels.tool import AbstractTool

logger = logging.getLogger(__name__)


@dataclass
class PlanPromptTemplate:
    """Used to generate or update the plan."""

    init_plan: str
    update_plan: str


@dataclass
class ActionPromptTemplate:
    """Used to generate the final answer or action."""

    system: str
    final_answer: str


@dataclass
class FinalAnswerPromptTemplate:
    """Used to generate the final answer if maximum steps achieved."""

    final_answer: str


@dataclass
class MultiStepPromptTemplate:
    plan: PlanPromptTemplate
    acter: ActionPromptTemplate
    final: FinalAnswerPromptTemplate


def _validate_prompt_template(prompt_template: MultiStepPromptTemplate) -> None:
    if not validate_template(prompt_template.plan.init_plan, {'task', 'tools'}):
        raise ValueError('plan.init_plan missing required fields {task, tools}')

    if not validate_template(
        prompt_template.plan.update_plan,
        {'task', 'tools', 'history', 'steps', 'remaining_steps'},
    ):
        raise ValueError(
            'plan.update_plan missing required fields {task, tools, history, steps, remaining_steps}'
        )

    if not validate_template(
        prompt_template.acter.final_answer,
        {'task', 'tools', 'history', 'steps', 'tool_calls'},
    ):
        raise ValueError(
            'acter.final_answer missing required fields {task, tools, history}'
        )

    if not validate_template(
        prompt_template.final.final_answer, {'task', 'history', 'steps'}
    ):
        raise ValueError(
            'final.final_answer missing required fields {task, tools, history, steps}'
        )


class TinyMultiStepAgent(AbstractAgent):
    def __init__(
        self,
        llm: AbstractLLM,
        prompt_template: MultiStepPromptTemplate,
        tools: list[AbstractTool] = [],
        memory_list: list[AbstractMemory] = [],
        max_steps: int = 15,
        plan_interval: int = 5,
    ) -> None:
        _validate_prompt_template(prompt_template)

        self._final_answer: str | None = None
        self._step_number: int = 1
        self._planned_steps: list[TinyPlanMessage] = []
        self._tool_calls: list[TinyToolCall] = []

        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.plan_interval = plan_interval

        self.acter_prompt = prompt_template.acter
        self.plan_prompt = prompt_template.plan
        self.final_prompt = prompt_template.final

        if not any(isinstance(m, BufferChatMemory) for m in memory_list):
            memory_list.append(BufferChatMemory())

        self.memory = MemoryGroup(memory_list=memory_list)

    @property
    def final_answer(self) -> str | None:
        return self._final_answer

    @final_answer.setter
    def final_answer(self, _: str | None) -> None:
        raise AttributeError('final_answer is read-only and cannot be set directly.')

    def _stream_steps(
        self, task: str
    ) -> Generator[TinyPlanMessage | TinyReasoningMessage]:
        class TinyReasonedSteps(TinyModel):
            planned_steps: list[str]
            reasoning: str

        # Initial plan
        if self._step_number == 1:
            messages = TinyLLMInput(
                messages=[
                    HumanMessage(
                        render_template(
                            self.plan_prompt.init_plan,
                            {'task': task, 'tools': self.tools},
                        )
                    )
                ]
            )
        else:
            messages = TinyLLMInput(
                messages=[
                    HumanMessage(
                        render_template(
                            self.plan_prompt.update_plan,
                            {
                                'task': task,
                                'tools': self.tools,
                                'history': self.memory.load_variables(),
                                'steps': self._planned_steps,
                                'remaining_steps': self.max_steps
                                - self._step_number
                                + 1,
                            },
                        )
                    )
                ]
            )

        result = self.llm.generate_structured(
            llm_input=messages,
            output_schema=TinyReasonedSteps,
        )

        yield TinyReasoningMessage(content=result.reasoning)
        for step in result.planned_steps:
            yield TinyPlanMessage(content=step)

    def _stream_action(self, task: str) -> Generator[TinyAIMessage]:
        messages = TinyLLMInput(
            messages=[
                SystemMessage(self.acter_prompt.system),
                HumanMessage(
                    render_template(
                        self.acter_prompt.final_answer,
                        {
                            'task': task,
                            'tools': self.tools,
                            'tool_calls': self._tool_calls,
                            'history': self.memory.load_variables(),
                            'steps': self._planned_steps,
                        },
                    )
                ),
            ]
        )

        action_tools = self.tools.copy()
        action_tools.append(provide_final_answer)

        result = self.llm.generate_with_tools(
            llm_input=messages,
            tools=action_tools,
        )
        for msg in result.tiny_iter():
            yield msg

    def _stream_final_answer(self, task: str) -> Generator[TinyChatMessage]:
        messages = TinyLLMInput(
            messages=[
                HumanMessage(
                    render_template(
                        self.final_prompt.final_answer,
                        {
                            'task': task,
                            'history': self.memory.load_variables(),
                            'steps': self._planned_steps,
                        },
                    )
                )
            ]
        )

        result = self.llm.generate_text(llm_input=messages)
        for msg in result.tiny_iter():
            if isinstance(msg, TinyChatMessage):
                yield msg

    def _run_generator(self, input_text: str) -> Generator[AllTinyMessages]:
        self._step_number = 1
        returned_final_answer: bool = False

        self.memory.save_context(TinyHumanMessage(content=input_text))

        while not returned_final_answer and (self._step_number <= self.max_steps):
            logger.info(f'--- STEP {self._step_number} ---')

            if self._step_number == 1 or (
                (self._step_number - 1) % self.plan_interval == 0
            ):
                # Create new plan
                plan_generator = self._stream_steps(input_text)
                self._planned_steps = []

                for msg in plan_generator:
                    if isinstance(msg, TinyPlanMessage):
                        logger.info(f'[{self._step_number}. STEP - Plan]: {msg.content}')
                        self._planned_steps.append(msg)
                    if isinstance(msg, TinyReasoningMessage):
                        logger.info(
                            f'[{self._step_number}. STEP - Reasoning]: {msg.content}'
                        )
                    self.memory.save_context(msg)

            try:
                for msg in self._stream_action(input_text):  # type: ignore
                    if isinstance(msg, TinyToolCall):
                        msg.call()
                        self._tool_calls.append(msg)

                        logger.info(
                            '[%s. STEP - Tool Call]: %s(%s) = %s',
                            self._step_number,
                            msg.tool_name,
                            msg.arguments,
                            msg.result,
                        )
                        if isinstance(msg.result, TinyChatMessage) and is_final_answer(
                            msg.result
                        ):
                            returned_final_answer = True
                            self._final_answer = msg.result.content
                    else:
                        logger.debug(
                            f'[{self._step_number}. STEP - Action]: {msg.tiny_str}'
                        )

                    self.memory.save_context(msg)
                    yield msg

                    if returned_final_answer:
                        break
            except Exception as e:
                raise e
            finally:
                self._step_number += 1

        if not returned_final_answer:
            logger.warning(
                'Max steps reached without returning a final answer. '
                'Returning the last known answer or a default message.'
            )

            final_answer_generator = self._stream_final_answer(input_text)
            for final_msg in final_answer_generator:
                self._final_answer = final_msg.content
                self.memory.save_context(final_msg)
                yield final_msg

    def run(
        self,
        input_text: str,
        reset: bool = True,
    ) -> str:
        logger.info(f'[USER INPUT] {input_text}')

        if reset:
            self._final_answer = None
            self._step_number = 1
            self._planned_steps = []
            self.memory.clear()

        results = list(self._run_generator(input_text))
        for res in results:
            logger.info(f'[AGENT OUTPUT] {res.tiny_str}')

        return self.final_answer or 'No final answer provided.'
