from collections.abc import Generator
from dataclasses import dataclass
import logging

from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from tinygent.datamodels.agent import AbstractAgent
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm_io import TinyLLMInput
from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.messages import AllTinyMessages
from tinygent.datamodels.messages import TinyAIMessage
from tinygent.datamodels.messages import TinyChatMessage
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.datamodels.messages import TinyPlanMessage
from tinygent.datamodels.messages import TinyToolCall
from tinygent.datamodels.tool import AbstractTool
from tinygent.memory.base_chat_memory import BaseChatMemory
from tinygent.tools.default import provide_final_answer
from tinygent.utils.jinja_utils import render_template

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


class TinyReActAgent(AbstractAgent):
    def __init__(
        self,
        llm: AbstractLLM,
        tools: list[AbstractTool] = [],
        memory: AbstractMemory = BaseChatMemory(),
        prompt_template: ReactPromptTemplate | None = None,
        max_steps: int = 15,
        plan_interval: int = 5,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.max_steps = max_steps
        self.plan_interval = plan_interval

        self.step_number: int = 1
        self.planned_steps: list[TinyPlanMessage] = []

        if prompt_template is None:
            self.plan_prompt = PlanPromptTemplate(init_plan='', update_plan='')
            self.acter_prompt = ActionPromptTemplate(system='', final_answer='')
        else:
            self.plan_prompt = prompt_template.plan
            self.acter_prompt = prompt_template.acter

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

                        if isinstance(
                            msg.result, TinyChatMessage
                        ) and msg.result.metadata.get('is_final_answer', False):
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
