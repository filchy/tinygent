from __future__ import annotations

from collections.abc import Generator
import logging
import typing
from typing import Any
from typing import Literal

from tinygent.agents import TinyBaseAgent
from tinygent.agents import TinyBaseAgentConfig
from tinygent.cli.builder import build_llm
from tinygent.cli.builder import build_tool
from tinygent.datamodels.llm_io_input import TinyLLMInput
from tinygent.datamodels.messages import TinyAIMessage
from tinygent.datamodels.messages import TinyChatMessage
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.datamodels.messages import TinyPlanMessage
from tinygent.datamodels.messages import TinyReasoningMessage
from tinygent.datamodels.messages import TinySystemMessage
from tinygent.datamodels.messages import TinyToolCall
from tinygent.datamodels.prompt import TinyPromptTemplate
from tinygent.memory import BufferChatMemory
from tinygent.tools.default_tools import provide_final_answer
from tinygent.tools.reasoning_tool import ReasoningTool
from tinygent.types.base import TinyModel
from tinygent.utils import is_final_answer
from tinygent.utils import render_template

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm import AbstractLLM
    from tinygent.datamodels.tool import AbstractTool

logger = logging.getLogger(__name__)


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


class FinalAnswerPromptTemplate(TinyPromptTemplate):
    """Used to generate the final answer if maximum steps achieved."""

    final_answer: str

    _template_fields = {
        'final_answer': {'task', 'history', 'steps'},
    }


class MultiStepPromptTemplate(TinyPromptTemplate):
    """Prompt templates for the multi-step agent."""

    plan: PlanPromptTemplate
    acter: ActionPromptTemplate
    final: FinalAnswerPromptTemplate


class TinyMultiStepAgentConfig(TinyBaseAgentConfig['TinyMultiStepAgent']):
    """Configuration for the TinyMultiStepAgent."""

    type: Literal['multistep'] = 'multistep'

    prompt_template: MultiStepPromptTemplate
    max_iterations: int = 15
    plan_interval: int = 5

    def build(self) -> TinyMultiStepAgent:
        return TinyMultiStepAgent(
            llm=build_llm(self.llm),
            prompt_template=self.prompt_template,
            tools=[build_tool(tool) for tool in self.tools],
            max_iterations=self.max_iterations,
            plan_interval=self.plan_interval,
        )


class TinyMultiStepAgent(TinyBaseAgent):
    """Multi-Step Agent implementation."""

    def __init__(
        self,
        llm: AbstractLLM,
        prompt_template: MultiStepPromptTemplate,
        tools: list[AbstractTool] = [],
        max_iterations: int = 15,
        plan_interval: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(llm=llm, tools=tools, **kwargs)

        self._step_number: int = 1
        self._planned_steps: list[TinyPlanMessage] = []
        self._tool_calls: list[TinyToolCall] = []

        __all_tools = list(tools) + [provide_final_answer]
        self._tools: list[AbstractTool] = [tool for tool in __all_tools]

        self.max_iterations = max_iterations
        self.plan_interval = plan_interval

        self.acter_prompt = prompt_template.acter
        self.plan_prompt = prompt_template.plan
        self.final_prompt = prompt_template.final

        self.memory = BufferChatMemory()

    def _stream_steps(
        self, task: str
    ) -> Generator[TinyPlanMessage | TinyReasoningMessage]:
        class TinyReasonedSteps(TinyModel):
            planned_steps: list[str]
            reasoning: str

        variables: dict[str, Any]

        # Initial plan
        if self._step_number == 1:
            template = self.plan_prompt.init_plan
            variables = {'task': task, 'tools': self.tools}
        else:
            template = self.plan_prompt.update_plan
            variables = {
                'task': task,
                'tools': self.tools,
                'history': self.memory.load_variables(),
                'steps': self._planned_steps,
                'remaining_steps': self.max_iterations - self._step_number + 1,
            }

        messages = TinyLLMInput(
            messages=[
                *self.memory.chat_messages,
                TinyHumanMessage(
                    content=render_template(
                        template,
                        variables,
                    )
                ),
            ]
        )

        result = self.run_llm(
            self.llm.generate_structured,
            llm_input=messages,
            output_schema=TinyReasonedSteps,
        )

        yield TinyReasoningMessage(content=result.reasoning)
        for step in result.planned_steps:
            yield TinyPlanMessage(content=step)

    def _stream_action(self, task: str) -> Generator[TinyAIMessage]:
        messages = TinyLLMInput(
            messages=[
                *self.memory.chat_messages,
                TinySystemMessage(content=self.acter_prompt.system),
                TinyHumanMessage(
                    content=render_template(
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

        result = self.run_llm(
            self.llm.generate_with_tools,
            llm_input=messages,
            tools=self._tools,
        )
        for msg in result.tiny_iter():
            yield msg

    def _stream_final_answer(self, task: str) -> Generator[TinyChatMessage]:
        messages = TinyLLMInput(
            messages=[
                *self.memory.chat_messages,
                TinyHumanMessage(
                    content=render_template(
                        self.final_prompt.final_answer,
                        {
                            'task': task,
                            'history': self.memory.load_variables(),
                            'steps': self._planned_steps,
                        },
                    )
                ),
            ]
        )

        result = self.run_llm(self.llm.generate_text, llm_input=messages)
        for msg in result.tiny_iter():
            if isinstance(msg, TinyChatMessage):
                yield msg

    def _run_agent(self, input_text: str) -> Generator[str]:
        self._step_number = 1
        returned_final_answer: bool = False

        self.memory.save_context(TinyHumanMessage(content=input_text))

        while not returned_final_answer and (self._step_number <= self.max_iterations):
            logger.debug('--- ITERATION %d ---', self._step_number)

            if self._step_number == 1 or (
                (self._step_number - 1) % self.plan_interval == 0
            ):
                # Create new plan
                plan_generator = self._stream_steps(input_text)
                self._planned_steps = []

                for msg in plan_generator:
                    if isinstance(msg, TinyPlanMessage):
                        logger.debug(
                            '[%d. ITERATION - Plan]: %s', self._step_number, msg.content
                        )
                        self.on_plan(msg.content)
                        self._planned_steps.append(msg)
                    if isinstance(msg, TinyReasoningMessage):
                        logger.debug(
                            '[%d. ITERATION - Reasoning]: %s',
                            self._step_number,
                            msg.content,
                        )
                        self.on_reasoning(msg.content)
                    self.memory.save_context(msg)

            try:
                for msg in self._stream_action(input_text):  # type: ignore
                    self.memory.save_context(msg)

                    if isinstance(msg, TinyChatMessage):
                        logger.debug(
                            '[%d. ITERATION - Chat]: %s', self._step_number, msg.content
                        )
                        returned_final_answer = True

                        self.on_answer(msg.content)
                        yield msg.content

                    elif isinstance(msg, TinyToolCall):
                        called_tool = self.get_tool(msg.tool_name)
                        if called_tool:
                            self.memory.save_context(self.run_tool(called_tool, msg))
                            self._tool_calls.append(msg)
                        else:
                            logger.error(
                                'Tool %s not found. Skipping tool call.', msg.tool_name
                            )

                        if isinstance(called_tool, ReasoningTool):
                            reasoning = msg.arguments.get('reasoning', '')
                            logger.debug(
                                '[%d. ITERATION - Tool Reasoning]: %s',
                                self._step_number,
                                reasoning,
                            )
                            self.on_tool_reasoning(reasoning)

                        logger.debug(
                            '[%s. ITERATION - Tool Call]: %s(%s) = %s',
                            self._step_number,
                            msg.tool_name,
                            msg.arguments,
                            msg.result,
                        )

                        if isinstance(msg.result, TinyChatMessage) and is_final_answer(
                            msg.result
                        ):
                            returned_final_answer = True

                            self.on_answer(msg.result.content)
                            yield msg.result.content

                    else:
                        logger.warning('Unhandled message type: %s', msg)

                    if returned_final_answer:
                        break
            except Exception as e:
                self.on_error(e)
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
                self.on_answer(final_msg.content)
                self.memory.save_context(final_msg)
                yield final_msg.content

    def run(
        self,
        input_text: str,
        reset: bool = True,
    ) -> str:
        logger.debug('[USER INPUT] %s', input_text)

        if reset:
            logger.debug('[AGENT RESET]')

            self._step_number = 1
            self._planned_steps = []
            self._tool_calls = []
            self.memory.clear()

        final_answer: str = ''
        for res in self._run_agent(input_text):
            final_answer += res

        return final_answer
