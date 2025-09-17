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
from tinygent.datamodels.messages import TinyToolCall
from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.runtime.memory_group import MemoryGroup
from tinygent.tools.default_tools import provide_final_answer
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
    init_plan: str
    update_plan: str


@dataclass
class ActionPromptTemplate:
    system: str
    final_answer: str


@dataclass
class ReactPromptTemplate:
    plan: PlanPromptTemplate
    acter: ActionPromptTemplate


def _validate_prompt_template(prompt_template: ReactPromptTemplate) -> None:
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
        prompt_template.acter.final_answer, {'task', 'tools', 'history'}
    ):
        raise ValueError(
            'acter.final_answer missing required fields {task, tools, history}'
        )


class TinyReActAgent(AbstractAgent):
    def __init__(
        self,
        llm: AbstractLLM,
        prompt_template: ReactPromptTemplate,
        tools: list[AbstractTool] = [],
        memory_list: list[AbstractMemory] = [],
        max_steps: int = 15,
        plan_interval: int = 5,
    ) -> None:
        _validate_prompt_template(prompt_template)

        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.plan_interval = plan_interval

        self.step_number: int = 1
        self.planned_steps: list[TinyPlanMessage] = []

        self.plan_prompt = prompt_template.plan
        self.acter_prompt = prompt_template.acter

        if not any(isinstance(m, BufferChatMemory) for m in memory_list):
            memory_list.append(BufferChatMemory())

        self.memory = MemoryGroup(
            memory_list=memory_list
        )

    def _stream_reason(self, task: str) -> Generator[TinyPlanMessage]:
        # Initial plan
        if self.step_number == 1:
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
                                'steps': self.planned_steps,
                                'remaining_steps': self.max_steps - self.step_number + 1,
                            },
                        )
                    )
                ]
            )

        result = self.llm.generate_with_tools(
            llm_input=messages,
            tools=self.tools,
        )

        for msg in result.tiny_iter():
            yield TinyPlanMessage(content=str(msg))

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
                            'history': self.memory.load_variables(),
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

    def _run_generator(self, input_text: str) -> Generator[AllTinyMessages]:
        self.step_number = 1
        returned_final_answer: bool = False

        self.memory.save_context(TinyHumanMessage(content=input_text))

        while not returned_final_answer and (self.step_number <= self.max_steps):
            logger.info(f'--- STEP {self.step_number} ---')

            if self.step_number == 1 or (
                (self.step_number - 1) % self.plan_interval == 0
            ):
                # Create new plan
                plan_generator = self._stream_reason(input_text)
                self.planned_steps = []

                for msg in plan_generator:
                    if isinstance(msg, TinyPlanMessage):
                        logger.info(f'[{self.step_number}. STEP - Plan]: {msg.content}')
                        self.planned_steps.append(msg)
                        self.memory.save_context(msg)

            try:
                for msg in self._stream_action(input_text):  # type: ignore
                    if isinstance(msg, TinyToolCall):
                        msg.call()

                        logger.info(
                            '[%s. STEP - Tool Call]: %s(%s) = %s',
                            self.step_number,
                            msg.tool_name,
                            msg.arguments,
                            msg.result,
                        )
                        if isinstance(msg.result, TinyChatMessage) and is_final_answer(
                            msg.result
                        ):
                            returned_final_answer = True

                    self.memory.save_context(msg)
                    yield msg

                    if returned_final_answer:
                        break
            except Exception as e:
                raise e
            finally:
                self.step_number += 1

    def run(
        self,
        input_text: str,
        reset: bool = True,
    ) -> str:
        logger.info(f'[USER INPUT] {input_text}')

        if reset:
            self.step_number = 1
            self.planned_steps = []
            self.memory.clear()

        results = list(self._run_generator(input_text))
        for res in results:
            logger.info(f'[AGENT OUTPUT] {res.tiny_str}')

        return str(results[-1])
